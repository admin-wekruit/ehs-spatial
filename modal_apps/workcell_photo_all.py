"""One ephemeral two A100 allocation for four original photos through RecGen models."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

import modal

if not modal.is_local():
    sys.path.insert(0, "/repo")
from modal_apps.fast_report_app import build_image

app = modal.App("workcell-photo-one-shot")
image = build_image(with_mapanything=True) if modal.is_local() else modal.Image.debian_slim()
volumes = {"/v/map": modal.Volume.from_name("mapanything-hf-cache"),
           "/v/sam3": modal.Volume.from_name("sam3-hf-cache"),
           "/v/models": modal.Volume.from_name("panoptes-fb-models"),
           "/cache": modal.Volume.from_name("panoptes-lucida-weights")}


def _start(command, gpu, *, hf=None):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = "/repo:/repo/scripts:/repo/modal_apps:/opt/recgen"
    env["HF_HUB_OFFLINE"] = "1"
    if hf:
        env["HF_HOME"] = hf
    env["ATTN_BACKEND"] = "xformers"
    env["SPCONV_ALGO"] = "native"
    return subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _finish(proc, label, timeout=1200):
    stdout, stderr = proc.communicate(timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"{label} failed: {(stdout + stderr)[-3000:]}")
    return stdout


@app.function(image=image, gpu="A100-80GB:2", cpu=16, memory=80 * 1024,
              volumes=volumes, timeout=1800, retries=0, min_containers=0)
def reconstruct(images: list[bytes], words: list[str], diameter_m: float, height_m: float):
    import numpy as np
    from PIL import Image
    import torch
    import trimesh
    from fast_report import coverage
    from scripts import workcell_photo_oneshot as report

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for i, payload in enumerate(images, 1):
            (root / f"source-{i}.jpg").write_bytes(payload)
        (root / "words.json").write_text(json.dumps(words))
        sources = [str(root / f"source-{i}.jpg") for i in range(1, 5)]
        map_proc = _start(["/opt/mapanything/bin/python", "/repo/scripts/workcell_map_worker.py", str(root), *sources], 0, hf="/v/map/huggingface")
        sam_proc = _start(["python", "/repo/scripts/workcell_sam_worker.py", str(root)], 1)
        active = []
        geometry_proc = None
        try:
            _finish(map_proc, "geometry")
            geometry_end = time.monotonic()
            frames = [np.asarray(Image.open(root / f"photo-{i}.png").convert("RGB"))[..., ::-1].copy() for i in range(1, 5)]
            worker = coverage.Detector("owlv2", torch.device("cuda:0"))
            rows = worker.detect(frames, coverage.SCORE["owlv2"])
            detections = [[{"box": box.tolist(), "score": round(float(score), 4), "word": word}
                           for box, score, word in zip(boxes, scores, names)] for boxes, scores, names in rows]
            del worker, rows
            torch.cuda.empty_cache()
            (root / "owl.json").write_text(json.dumps({"results": detections}))
            owl_end = time.monotonic()
            picks = []
            for row in detections:
                options = [x for x in row if x["word"] in ("cart", "work platform")]
                if not options:
                    raise ValueError("OWLv2 found no cart proposal")
                picks.append(max(options, key=lambda x: x["score"]))
            (root / "cart-boxes.json").write_text(json.dumps({"results": [[x] for x in picks]}))
            _finish(sam_proc, "SAM 3")
            sam_end = time.monotonic()
            seg = json.loads((root / "sam3.json").read_text())
            cart = json.loads((root / "cart-masks.json").read_text())
            report._prepare_inputs(root, seg, cart)
            prepare_end = time.monotonic()
            geometry_proc = _start([
                "python", "-c",
                "import sys; from pathlib import Path; from scripts.workcell_photo_geometry import build; "
                "build(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:6]], float(sys.argv[6]), float(sys.argv[7]))",
                str(root), *sources, str(diameter_m), str(height_m)], "")
            plans = ((0, "robot", "v1:1;v2:2;v3:3;v4:4;multi:1,2,3"), (1, "cart", "single:1"))
            for gpu, kind, groups in plans:
                source = root / f"{kind}-input.npz"
                target = root / kind
                proc = _start(["/opt/recgen-venv/bin/python", "/repo/scripts/workcell_recgen_worker.py", str(source), str(target), groups], gpu)
                active.append((kind, source, target, proc))
            timing = {}
            for kind, source, target, proc in active:
                output = _finish(proc, f"{kind} RecGen")
                worker = json.loads(output.strip().splitlines()[-1])
                z = np.load(source)
                for name, stats in worker["models"].items():
                    mesh_data = np.load(f"{target}-{name}.npz")
                    mesh = trimesh.Trimesh(vertices=mesh_data["vertices"], faces=mesh_data["faces"],
                                           vertex_colors=mesh_data["colors"], process=False)
                    mesh.apply_transform(z[f"v{stats['views'][0]}_c2w"])
                    (root / f"{kind}-{name}.glb").write_bytes(mesh.export(file_type="glb"))
                    timing[f"{kind}-{name}"] = {**stats, "modelLoadSeconds": worker["modelLoadSeconds"]}
            model_end = time.monotonic()
            _finish(geometry_proc, "metric geometry")
            complete_end = time.monotonic()
            (root / "models-timing.json").write_text(json.dumps({"models": timing, "containerWallSeconds": model_end - started}))
            (root / "stage-timing.json").write_text(json.dumps({"geometrySeconds": geometry_end-started,
                "owlSeconds": owl_end-geometry_end, "segmentationSeconds": json.loads((root / "sam-timing.json").read_text())["containerSeconds"],
                "cartMaskSeconds": sam_end-owl_end, "prepareSeconds": prepare_end-sam_end,
                "modelSeconds": model_end-prepare_end,
                "metricTailSeconds": complete_end-model_end}))
            wanted = [*root.glob("frame_*.json.gz"), *root.glob("photo-*.png"), *root.glob("*.json"),
                      *root.glob("*-input.npz"), *root.glob("*.glb"), *root.glob("geometry-*.jpg"),
                      *root.glob("geometry-*.png")]
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w:gz") as archive:
                for path in wanted:
                    if path.name.startswith("source-") or path.name == "words.json":
                        continue
                    archive.add(path, arcname=path.name)
            return {"archive": payload.getvalue(), "containerWallSeconds": time.monotonic() - started}
        finally:
            for proc in (map_proc, sam_proc, geometry_proc, *(job[3] for job in active)):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)


@app.local_entrypoint()
def main(images: str, out: str, words: str, button_diameter_m: float = .2, button_height_m: float = .2):
    started = time.monotonic()
    paths = [Path(p) for p in images.split(",")]
    if len(paths) != 4 or any(not p.is_file() for p in paths):
        raise ValueError("Exactly four source photos required")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    import math
    if not all(math.isfinite(v) and v > 0 for v in (button_diameter_m, button_height_m)):
        raise ValueError("Button dimensions must be finite and positive")
    result = reconstruct.remote([p.read_bytes() for p in paths], words.split(","), button_diameter_m, button_height_m)
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as archive:
        for member in archive.getmembers():
            if Path(member.name).name != member.name:
                raise ValueError("Unexpected archive path")
        archive.extractall(destination, filter="data")
    (destination / "modal-timing.json").write_text(json.dumps({"wallSecondsIncludingColdStart": time.monotonic() - started,
                                                             "containerWallSeconds": result["containerWallSeconds"]}))
    print(json.dumps({"status": "ok", "wallSecondsIncludingColdStart": time.monotonic() - started}))
