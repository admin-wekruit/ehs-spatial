"""One ephemeral two A100 allocation for four original photos through RecGen models."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tarfile
import tempfile
import time

import modal

if not modal.is_local():
    sys.path[:0] = ["/repo", "/repo/scripts"]
from modal_apps.fast_report_app import build_image
from fast_report.sam3d import TORCH_HUB

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
    env["TORCH_HOME"] = TORCH_HUB  # Reuse image-baked DINOv2; parallel Hub bootstrap mutates one shared cache.
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



def _archive_result(root, started, error=None):
    if error is not None:
        (root / "failure.json").write_text(json.dumps(error))
    wanted = [*root.glob("frame_*.json.gz"), *root.glob("photo-*.png"), *root.glob("*.json"),
              *root.glob("*-input.npz"), *root.glob("*.glb"), *root.glob("geometry-*.jpg"),
              *root.glob("geometry-*.png"), *root.glob("raw-image-features-*.jpg"), *root.glob("structural-*.jpg")]
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in wanted:
            if path.name.startswith("source-") or path.name == "words.json":
                continue
            archive.add(path, arcname=path.name)
    return {"archive": payload.getvalue(), "containerWallSeconds": time.monotonic() - started, "error": error}

@app.function(image=image, gpu="A100-80GB:2", cpu=16, memory=80 * 1024,
              volumes=volumes, timeout=1800, retries=0, min_containers=0)
def reconstruct(images: list[bytes], words: list[str], diameter_m: float, height_m: float, reference: dict | None = None):
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
        (root / "reference-input.json").write_text(json.dumps(reference))
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
                picks.append(options)
            (root / "cart-boxes.json").write_text(json.dumps({"results": picks}))
            _finish(sam_proc, "SAM 3")
            sam_end = time.monotonic()
            seg = json.loads((root / "sam3.json").read_text())
            cart = json.loads((root / "cart-masks.json").read_text())
            geometry_proc = _start([
                "python", "-c",
                "import sys,json; from pathlib import Path; from scripts.workcell_photo_geometry import build; "
                "build(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:6]], float(sys.argv[6]), float(sys.argv[7]), reference=json.loads((Path(sys.argv[1])/'reference-input.json').read_text()))",
                str(root), *sources, str(diameter_m), str(height_m)], "")
            while not (root / "floor-reference.json").exists():
                if geometry_proc.poll() is not None:
                    _finish(geometry_proc, "floor reference")
                    raise ValueError("Geometry completed without its ground reference")
                time.sleep(.05)
            report._prepare_inputs(root, seg, cart)
            prepare_end = time.monotonic()
            guard_views = report._guard_views(json.loads((root / "guard-mask-selection.json").read_text()))
            if len(guard_views) < 2:
                raise ValueError("V-guard lacks independent support in at least two views")
            plans = [[("robot", [["v1",[1]],["v2",[2]],["v3",[3]],["v4",[4]],["multi",[1,2,3]]])],
                     [("cart", [["single",[1]]]), ("guard", [["multi",guard_views]])]]
            for gpu, jobs in enumerate(plans):
                plan = [{"kind":kind, "source":str(root/f"{kind}-input.npz"), "target":str(root/kind), "groups":groups} for kind, groups in jobs]
                plan_path = root / f"recgen-plan-{gpu}.json"
                plan_path.write_text(json.dumps(plan))
                proc = _start(["/opt/recgen-venv/bin/python", "/repo/scripts/workcell_recgen_worker.py", str(plan_path)], gpu)
                active.append((gpu, proc))
            timing = {}
            for gpu, proc in active:
                output = _finish(proc, f"GPU {gpu} RecGen")
                worker = json.loads(output.strip().splitlines()[-1])
                for kind, job in worker["jobs"].items():
                    z = np.load(root / f"{kind}-input.npz")
                    for name, stats in job["models"].items():
                        mesh_data = np.load(root / f"{kind}-{name}.npz")
                        mesh = trimesh.Trimesh(vertices=mesh_data["vertices"], faces=mesh_data["faces"],
                                               vertex_colors=mesh_data["colors"], process=False)
                        mesh.apply_transform(z[f"v{stats['views'][0]}_c2w"])
                        if kind == "guard":
                            from fast_report.x7 import light, rays, refine, Caster, score_view
                            v, f, _ = light(mesh.vertices, mesh.faces, mesh.visual.vertex_colors[:, :3])
                            views = []
                            for i in range(1, 5):
                                depth = z[f"v{i}_depth"][::2, ::2]
                                K = z[f"v{i}_K"].copy(); K[:2] /= 2
                                views.append({"rays": rays(K, z[f"v{i}_c2w"], depth.shape[1], depth.shape[0]),
                                              "target": z[f"v{i}_mask"][::2, ::2] > 0, "depth": depth})
                            transform, placement = refine(v, f, [views[i-1] for i in stats['views']], uniform_scale=True)
                            mesh.apply_transform(transform)
                            caster = Caster(v, f)
                            placement.update(generationViews=stats['views'], transform=transform.tolist(),
                                             sourceChecks=[score_view(caster, transform, view) for view in views],
                                             basis="shape-preserving similarity alignment to source masks and estimated depth; not surveyed physical accuracy")
                            (root / "guard-placement.json").write_text(json.dumps(placement, indent=2))
                        (root / f"{kind}-{name}.glb").write_bytes(mesh.export(file_type="glb"))
                        timing[f"{kind}-{name}"] = {**stats, "modelLoadSeconds": worker["modelLoadSeconds"]}
            model_end = time.monotonic()
            _finish(geometry_proc, "metric geometry")
            from scripts.workcell_photo_objects import build as build_objects
            from scripts.workcell_photo_report import build as build_report
            report._posts(root, seg)
            catalog = build_objects(root, [Path(path) for path in sources])
            from scripts.workcell_photo_metrology import apply_source_clearances
            from scripts.workcell_guard_silhouette import run as fit_shared_guards
            from scripts.workcell_guard_experiment_report import structural_models
            from concurrent.futures import ThreadPoolExecutor
            # Independent native-world fits share fixed cameras; neither consumes evaluation targets.
            (root / 'a1').mkdir()
            for side in ('left', 'right'):
                shutil.copy2(root/f'guard-{side}.glb', root/'a1'/f'guard-{side}.glb')
            with ThreadPoolExecutor(max_workers=2) as pool:
                physical = pool.submit(apply_source_clearances, root, [Path(path) for path in sources])
                structural = pool.submit(fit_shared_guards, root, root/'a4', [Path(path) for path in sources])
                physical.result(); structural.result()
            _, structural_result = structural_models(root/'a4', root, catalog)
            initializers = {}
            for side in ('left', 'right'):
                name = f'guard-{side}-initializer.glb'
                shutil.copy2(root/'a1'/f'guard-{side}.glb', root/name)
                initializers[side] = name
            partition = json.loads((root/'guard-partition.json').read_text())
            if 'initializerProvenance' in partition:
                partition['initializerProvenance']['models'] = initializers
                (root/'guard-partition.json').write_text(json.dumps(partition, indent=2))
            for i, name in enumerate(structural_result.get('overlays', [])):
                exported = 'structural-' + Path(name).name
                shutil.copy2(root/'a4'/name, root/exported)
                structural_result['overlays'][i] = exported
            structural_result['initializerModels'] = initializers
            (root/'structural-result.json').write_text(json.dumps(structural_result, indent=2))
            (root/'objects.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
            if reference:
                from scripts.workcell_photo_calibration import apply_measurements
                apply_measurements(root, {'schemaVersion': 1, 'reference': reference})
            build_report(root)
            complete_end = time.monotonic()
            (root / "models-timing.json").write_text(json.dumps({"models": timing, "containerWallSeconds": model_end - started}))
            (root / "stage-timing.json").write_text(json.dumps({"geometrySeconds": geometry_end-started,
                "owlSeconds": owl_end-geometry_end, "segmentationSeconds": json.loads((root / "sam-timing.json").read_text())["containerSeconds"],
                "cartMaskSeconds": sam_end-owl_end, "prepareSeconds": prepare_end-sam_end,
                "modelSeconds": model_end-prepare_end,
                "metricTailSeconds": complete_end-model_end}))
            return _archive_result(root, started)
        except Exception as error:
            return _archive_result(root, started, {'type': type(error).__name__, 'message': str(error)})
        finally:
            for proc in (map_proc, sam_proc, geometry_proc, *(job[1] for job in active)):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()


@app.local_entrypoint()
def main(images: str, out: str, words: str, button_diameter_m: float = .2, button_height_m: float = .2, reference: str = ""):
    started = time.monotonic()
    paths = [Path(p) for p in images.split(",")]
    if len(paths) != 4 or any(not p.is_file() for p in paths):
        raise ValueError("Exactly four source photos required")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    import math
    if not all(math.isfinite(v) and v > 0 for v in (button_diameter_m, button_height_m)):
        raise ValueError("Button dimensions must be finite and positive")
    reference_data = json.loads(Path(reference).read_text()) if reference else None
    if reference_data is not None:
        from scripts.workcell_photo_calibration import validate_measurements
        reference_data = validate_measurements({'schemaVersion': 1, 'reference': reference_data})['reference']
    result = reconstruct.remote([p.read_bytes() for p in paths], words.split(","), button_diameter_m, button_height_m, reference_data)
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as archive:
        for member in archive.getmembers():
            if Path(member.name).name != member.name:
                raise ValueError("Unexpected archive path")
        archive.extractall(destination, filter="data")
    (destination / "modal-timing.json").write_text(json.dumps({"wallSecondsIncludingColdStart": time.monotonic() - started,
                                                             "containerWallSeconds": result["containerWallSeconds"]}))
    if result.get('error'):
        raise RuntimeError(f"Cloud stage failed; completed artifacts retained: {result['error']}")
    print(json.dumps({"status": "ok", "wallSecondsIncludingColdStart": time.monotonic() - started}))
