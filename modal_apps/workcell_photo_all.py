"""One ephemeral two A100 allocation: N >= 2 original photos of one scene through one RecGen model per object."""
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

def _small_objects(root, catalog, photo_count, sources):
    """RecGen models for signal lamps (scripts/workcell_recgen_objects): accepted ones replace the proxy model.

    An optional display layer: any failure (inputs, a worker, placement) keeps every proxy, is recorded in
    coverage.smallObjectModels and never fails the paid run; the catalog changes only after every lamp is placed."""
    import numpy as np
    import trimesh
    from scripts import workcell_recgen_objects as small
    ids = [item['id'] for item in catalog['objects'] if item['kind'] == 'signal light' and item.get('model')]
    if not ids:
        return catalog
    procs = []
    try:
        jobs = small.build_inputs(root, range(1, photo_count + 1), ids, sources=[Path(p) for p in sources], scene='scene')
        for job in jobs:
            if job['record']['views']:
                np.savez_compressed(root / job['record']['input'], **job['arrays'])
        records = [job['record'] for job in jobs]
        for gpu, plan in enumerate(small.plans(records)):
            path = root / f'small-objects-plan-{gpu}.json'
            path.write_text(json.dumps(plan))
            procs.append(_start(["/opt/recgen-venv/bin/python", "/repo/scripts/workcell_recgen_worker.py", str(path)], gpu))
        workers = [json.loads(_finish(proc, f"GPU {gpu} small-object RecGen").strip().splitlines()[-1]) for gpu, proc in enumerate(procs)]
        for record in records:
            if record['views']:
                worker = next(w for w in workers if record['kind'] in w['jobs'])
                small.place(root, record, worker['jobs'][record['kind']]['models'][small.GROUP], worker['modelLoadSeconds'])
            else:
                (root / f"{record['kind']}.json").write_text(json.dumps(record, indent=2))
        nodes = {r['objectId']: list(trimesh.load(root / r['glb'], force='scene').graph.nodes_geometry) for r in records if r.get('accepted')}
    except Exception as error:  # ponytail: lamps are optional; the proxies stay and the reason is published
        catalog['coverage']['smallObjectModels'] = {'status': 'failed', 'proxiesKept': True, 'error': f'{type(error).__name__}: {error}'[-1500:]}
        return catalog
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    items = {item['id']: item for item in catalog['objects']}
    for record in records:
        item = items[record['objectId']]
        item['recgenModel'] = {key: record.get(key) for key in small.SUMMARY_KEYS}
        if record.get('accepted'):
            item['model'] = {'file': record['glb'], 'nodes': nodes[record['objectId']]}
            item['representation'] = 'RecGen model from every view of this lamp, placed by a multi-view similarity fit against every photo'
            item.pop('proxyModel', None)
            item.pop('modelDimensionsNative', None)  # the proxy's box size; the RecGen model's extent is recgenModel.modelExtentNative
            item['notes'] = ['One RecGen model per lamp; overexposed lamps make pale colours; stood upright on the fitted floor normal when every view still passes (recgenModel.upright).']
    catalog['coverage']['smallObjectModels'] = [{key: record.get(key) for key in ('objectId', 'views', 'accepted', 'reasons')} for record in records]
    return catalog


@app.function(image=image, gpu="A100-80GB:2", cpu=16, memory=80 * 1024,
              volumes=volumes, timeout=1800, retries=0, min_containers=0)
def reconstruct(images: list[bytes], words: list[str], diameter_m: float, height_m: float, reference: dict | None = None,
                capture: dict | None = None):
    import numpy as np
    from PIL import Image
    import torch
    import trimesh
    from fast_report import coverage
    from scripts import workcell_photo_oneshot as report

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        if capture is None or capture.get("photoCount") != len(images) or len(images) < 2:
            raise ValueError("A capture record naming every one of the >= 2 uploaded photos is required")
        (root / report.CAPTURE).write_text(json.dumps(capture, indent=2) + "\n")
        for i, payload in enumerate(images, 1):
            (root / f"source-{i}.jpg").write_bytes(payload)
        (root / "words.json").write_text(json.dumps(words))
        (root / "reference-input.json").write_text(json.dumps(reference))
        sources = [str(root / f"source-{i}.jpg") for i in range(1, len(images) + 1)]
        map_proc = _start(["/opt/mapanything/bin/python", "/repo/scripts/workcell_map_worker.py", str(root), *sources], 0, hf="/v/map/huggingface")
        sam_proc = _start(["python", "/repo/scripts/workcell_sam_worker.py", str(root)], 1)
        active = []
        geometry_proc = None
        try:
            _finish(map_proc, "geometry")
            geometry_end = time.monotonic()
            frames = [np.asarray(Image.open(root / f"photo-{i}.png").convert("RGB"))[..., ::-1].copy() for i in range(1, len(images) + 1)]
            worker = coverage.Detector("owlv2", torch.device("cuda:0"))
            rows = worker.detect(frames, coverage.SCORE["owlv2"])
            # OWLv2 is this process's first CUDA user, so the device peak is its peak (stats need an initialized device).
            owl_peak = torch.cuda.max_memory_allocated(0) / 2**30 if torch.cuda.is_initialized() else None
            detections = [[{"box": box.tolist(), "score": round(float(score), 4), "word": word}
                           for box, score, word in zip(boxes, scores, names)] for boxes, scores, names in rows]
            del worker, rows
            torch.cuda.empty_cache()
            (root / "owl.json").write_text(json.dumps({"results": detections}))
            owl_end = time.monotonic()
            # A view without a proposal simply shows no cart; at least one view must.
            picks = [[x for x in row if x["word"] in ("cart", "work platform")] for row in detections]
            if not any(picks):
                raise ValueError("OWLv2 found no cart proposal in any photo")
            (root / "cart-boxes.json").write_text(json.dumps({"results": picks}))
            _finish(sam_proc, "SAM 3")
            sam_end = time.monotonic()
            seg = json.loads((root / "sam3.json").read_text())
            cart = json.loads((root / "cart-masks.json").read_text())
            geometry_proc = _start([
                "python", "-c",
                "import sys,json; from pathlib import Path; from scripts.workcell_photo_geometry import build; "
                "build(Path(sys.argv[1]), [Path(p) for p in sys.argv[4:]], float(sys.argv[2]), float(sys.argv[3]), reference=json.loads((Path(sys.argv[1])/'reference-input.json').read_text()))",
                str(root), str(diameter_m), str(height_m), *sources], "")
            while not (root / "floor-reference.json").exists():
                if geometry_proc.poll() is not None:
                    _finish(geometry_proc, "floor reference")
                    raise ValueError("Geometry completed without its ground reference")
                time.sleep(.05)
            report._prepare_inputs(root, seg, cart)
            prepare_end = time.monotonic()
            # One model per object: robot and cart from every view that shows them, the guard from its two best views.
            plans, refine_views = report.recgen_plans(root)
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
                        # The same Sim(3) refine for every model, against the masks and depth of its placement views.
                        mesh, placement = report.place_model(mesh, z, stats["views"], refine_views[kind])
                        (root / f"{kind}-placement.json").write_text(json.dumps(placement, indent=2))
                        (root / f"{kind}-{name}.glb").write_bytes(mesh.export(file_type="glb"))
                        timing[f"{kind}-{name}"] = {**stats, "modelLoadSeconds": worker["modelLoadSeconds"],
                                                    "peakAllocatedGiB": worker.get("peakAllocatedGiB"),
                                                    "placementSeconds": placement["seconds"]}
            model_end = time.monotonic()
            _finish(geometry_proc, "metric geometry")
            from scripts.workcell_photo_objects import build as build_objects
            from scripts.workcell_photo_report import finalize
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
            # Physical-edge fitting can replace the ground plane. Refresh models
            # and their existing catalog together before integrating A4 geometry.
            report._posts(root, seg)
            catalog = build_objects(root, [Path(path) for path in sources])
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
            # Signal lamps: one RecGen model per lamp from all its views, placed against every photo (both GPUs idle here).
            lamps_started = time.monotonic()
            catalog = _small_objects(root, catalog, len(images), sources)
            lamp_seconds = time.monotonic() - lamps_started
            (root/'objects.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2))
            # Models, catalog and floor are final here; no _posts/build_objects may follow.
            finalize(root, {'schemaVersion': 1, 'reference': reference} if reference else None)
            complete_end = time.monotonic()
            (root / "models-timing.json").write_text(json.dumps({"models": timing, "containerWallSeconds": model_end - started}))
            (root / "stage-timing.json").write_text(json.dumps({"photoCount": len(images), "geometrySeconds": geometry_end-started,
                "owlSeconds": owl_end-geometry_end, "owlPeakAllocatedGiB": owl_peak,
                "segmentationSeconds": json.loads((root / "sam-timing.json").read_text())["containerSeconds"],
                "cartMaskSeconds": sam_end-owl_end, "prepareSeconds": prepare_end-sam_end,
                "modelSeconds": model_end-prepare_end,
                "metricTailSeconds": complete_end-model_end, "smallObjectSeconds": lamp_seconds}))
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
def main(images: str, out: str, words: str, button_diameter_m: float = .2, button_height_m: float = .2, reference: str = "",
         reference_photo: int = 0):
    from scripts.workcell_photo_oneshot import capture_record
    started = time.monotonic()
    paths = [Path(p) for p in images.split(",")]
    capture = capture_record(paths, reference_photo)  # >= 2 distinct readable photos of one scene; 0 = the last
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    import math
    if not all(math.isfinite(v) and v > 0 for v in (button_diameter_m, button_height_m)):
        raise ValueError("Button dimensions must be finite and positive")
    reference_data = json.loads(Path(reference).read_text()) if reference else None
    if reference_data is not None:
        from scripts.workcell_photo_calibration import validate_measurements
        reference_data = validate_measurements({'schemaVersion': 1, 'reference': reference_data})['reference']
    result = reconstruct.remote([p.read_bytes() for p in paths], words.split(","), button_diameter_m, button_height_m, reference_data, capture)
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
