"""Bounded four-photo LingBot camera/depth control, not an accepted metric report.

Run once with ``python -m modal run modal_apps/workcell_lingbot_probe.py
--images-json FOUR_PATHS_JSON --baseline RUN_B --measurements CONFIG --out NEW_DIR``.
Only original RGB, frozen raw button observations and reference dimensions enter
the GPU container. Ground-truth clearances and old camera/depth geometry do not.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'modal_apps')]
from lingbot_room import image as base_image, volume, REV, WEIGHTS_REV, WEIGHTS_SHA, digest

app = modal.App('workcell-lingbot-photo-probe')
SCRIPTS = ('workcell_button_bundle.py', 'workcell_photo_metrology.py',
           'workcell_photo_geometry.py', 'workcell_photo_objects.py',
           'workcell_photo_oneshot.py', 'workcell_guard_joint.py', 'workcell_metrology_models.py')
image = base_image.pip_install('trimesh==5.1.0', 'pydantic==2.12.3', 'Pillow==12.0.0')
if modal.is_local():
    for name in SCRIPTS:
        image = image.add_local_file(ROOT / 'scripts' / name, '/repo/scripts/' + name)
    for path in sorted((ROOT / 'ehs_spatial').rglob('*.py')):
        image = image.add_local_file(path, '/repo/' + str(path.relative_to(ROOT)))
    image = image.add_local_file(Path(__file__), '/repo/modal_apps/workcell_lingbot_probe.py')
    image = image.add_local_file(ROOT / 'modal_apps/lingbot_room.py', '/repo/modal_apps/lingbot_room.py')


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def canonical_image(path, target=518, patch=14):
    """Official crop raster + pixel-center affine; runtime compares every pixel."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError('Frozen observations require upright input pixels; EXIF rotation unsupported')
        source = source.convert('RGB')
        width, height = source.size
        resized_height = round(height * target / width / patch) * patch
        crop_y = max(0, (resized_height - target) // 2)
        rgb = np.asarray(source.resize((target, resized_height), Image.Resampling.BICUBIC))
        rgb = rgb[crop_y:crop_y + min(resized_height, target)]
    sx, sy = target / width, resized_height / height
    A = np.array([[sx, 0., (sx - 1.) / 2], [0., sy, (sy - 1.) / 2 - crop_y], [0., 0., 1.]])
    return rgb, A, (width, height)


def frozen_inputs(baseline, measurements):
    from workcell_photo_metrology import _reference
    fit = json.loads((Path(baseline) / 'geometry.json').read_text())['anchor']['referenceFit']
    observations = fit['observations']
    expected = fit['sourceContourSha256']
    if hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest() != expected:
        raise ValueError('Frozen source contours changed')
    reference = _reference(json.loads(Path(measurements).read_text()))
    if reference != _reference(fit['reference']):
        raise ValueError('Reference differs from baseline; this is not a same-input comparison')
    timing = json.loads((Path(baseline) / 'geometry-timing.json').read_text())
    hashes = [row['sourceSha256'] for row in timing['frameSummaries']]
    if len(hashes) != 4:
        raise ValueError('Baseline lacks four bound original image hashes')
    return {'observations': observations, 'sourceContourSha256': expected, 'reference': reference,
            'sourceSha256': hashes}


def export_surface(frames, out):
    """Small observed-surface diagnostic; no hole completion or metric claims."""
    import numpy as np
    import trimesh
    scene = trimesh.Scene()
    for photo, frame in frames.items():
        # ponytail: quarter-grid surface caps delivery size; full depth stays on the volume.
        points, depth = frame['points'][::4, ::4], frame['depth'][::4, ::4]
        colors = frame['canonicalRgb'][::4, ::4]
        h, w = depth.shape
        ids = np.arange(h * w).reshape(h, w)
        faces = np.concatenate([np.stack([ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:]], -1).reshape(-1, 3),
                                np.stack([ids[1:, 1:], ids[:-1, 1:], ids[1:, :-1]], -1).reshape(-1, 3)])
        valid = np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0)
        z = depth.ravel()[faces]
        keep = valid.ravel()[faces].all(-1) & (np.ptp(z, axis=1) / np.maximum(np.min(z, axis=1), 1e-8) < .08)
        mesh = trimesh.Trimesh(vertices=np.nan_to_num(points.reshape(-1, 3)), faces=faces[keep],
                               vertex_colors=colors.reshape(-1, 3), process=False)
        mesh.remove_unreferenced_vertices()
        scene.add_geometry(mesh, node_name=f'observed-photo-{photo}')
    (out / 'observed-native.glb').write_bytes(scene.export(file_type='glb'))


def fit_and_overlay(frames, frozen, out):
    import cv2
    import numpy as np
    import trimesh
    from workcell_button_bundle import fit_reference_shape, _project_raw
    from workcell_metrology_models import reference_meshes
    # Only an initialization: camera-up is not a measured floor normal.
    axis = -np.mean([f['pose'][:3, 1] for f in frames.values()], axis=0)
    axis /= np.linalg.norm(axis)
    fit = fit_reference_shape(frames, frozen['observations'], frozen['reference'], axis)
    fit['gauge']['worldFrame'] = 'LingBot native'
    save(out / 'reference-fit.json', fit)
    meshes = reference_meshes(fit['fittedNuisanceParameters'])
    scene = trimesh.Scene(meshes)
    (out / 'button-candidate-native.glb').write_bytes(scene.export(file_type='glb'))
    for row in frozen['observations']:
        photo = row['photo']; frame = frames[photo]
        box = np.asarray(row['boxRaw'], int)
        pad = max(80, int(max(box[2:] - box[:2])))
        x0, y0 = np.maximum(box[:2] - pad, 0)
        x1, y1 = np.minimum(box[2:] + pad, frame['rgb'].shape[:2][::-1])
        crop = frame['rgb'][y0:y1, x0:x1, ::-1].copy()
        for key in ('redHullRaw', 'yellowHullRaw'):
            uv = np.asarray(row[key]) - [x0, y0]
            cv2.polylines(crop, [np.rint(uv).astype(np.int32)], True, (255, 220, 0), 2)
        for mesh in meshes.values():
            uv, depths = _project_raw(mesh.vertices, frame)
            if (depths <= 0).any():
                continue
            hull = cv2.convexHull(np.rint(uv - [x0, y0]).astype(np.int32))
            cv2.polylines(crop, [hull], True, (0, 165, 255), 2)
        cv2.imwrite(str(out / f'button-overlay-{photo}.jpg'), crop)
    return fit


@app.function(image=image, gpu='A100-80GB:2', cpu=16, memory=80 * 1024,
              volumes={'/artifact': volume}, timeout=1200, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2)
def probe(payloads: list[bytes], frozen: dict, run_id: str):
    started = time.monotonic()
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    sys.path[:0] = ['/repo', '/repo/scripts', '/opt/lingbot']
    from demo import load_model
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from types import SimpleNamespace
    torch.manual_seed(0)
    np.random.seed(0)
    volume.reload()
    if not run_id or any(not (c.isalnum() or c in '-_') for c in run_id):
        raise ValueError('Invalid unique run ID')
    root = Path('/artifact') / run_id
    root.mkdir(exist_ok=False)
    out = root / 'result'; out.mkdir()
    error = None; stats = {}
    try:
        if len(payloads) != 4 or len({hashlib.sha256(b).hexdigest() for b in payloads}) != 4:
            raise ValueError('Exactly four distinct original JPEGs required')
        if [hashlib.sha256(b).hexdigest() for b in payloads] != frozen['sourceSha256']:
            raise ValueError('Images or their order differ from frozen source observations')
        save(root / 'frozen-inputs.json', frozen)
        paths = []
        for i, payload in enumerate(payloads, 1):
            path = root / f'input-{i}.jpg'; path.write_bytes(payload); paths.append(path)
        prepared = [canonical_image(p) for p in paths]
        tensor = load_and_preprocess_images([str(p) for p in paths], mode='crop', image_size=518, patch_size=14)
        for i, (rgb, _, _) in enumerate(prepared):
            actual = tensor[i].permute(1, 2, 0).cpu().numpy()
            if actual.shape != rgb.shape or np.max(np.abs(actual - rgb / 255.)) > 1e-6:
                raise ValueError('Canonical crop does not match pinned official loader')
        weight = hf_hub_download('robbyant/lingbot-map', 'lingbot-map.pt', revision=WEIGHTS_REV,
                                 cache_dir='/artifact/hf', local_files_only=True)
        if digest(weight) != WEIGHTS_SHA:
            raise ValueError('Cached LingBot weight checksum mismatch')
        args = SimpleNamespace(mode='streaming', image_size=518, patch_size=14, enable_3d_rope=True,
            max_frame_num=1024, kv_cache_sliding_window=64, num_scale_frames=4, use_sdpa=True,
            camera_num_iterations=4, model_path=weight)
        model = load_model(args, 'cuda:0'); model.aggregator = model.aggregator.to(dtype=torch.bfloat16); model.eval()
        torch.cuda.reset_peak_memory_stats(); began = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model.inference_streaming(tensor.to('cuda:0'), num_scale_frames=4,
                                                  keyframe_interval=1, output_device=torch.device('cpu'))
        torch.cuda.synchronize()
        stats = {'inferenceSeconds': time.monotonic() - began,
                 'peakGpuAllocatedGiB': torch.cuda.max_memory_allocated() / 2**30,
                 'gpu': torch.cuda.get_device_name(0), 'gpuCountReserved': 2, 'gpuCountUsedByModel': 1}
        # Save before all camera/mesh/fit adaptation so failure never requires another forward pass.
        torch.save(prediction, out / 'native-prediction.pt')
        save(out / 'native-prediction.json', {**stats, 'sha256': digest(out / 'native-prediction.pt')})
        volume.commit()
        # Decode native pose directly. Pinned demo.postprocess already inverts
        # W2C to C2W; calling it and inverting again would flip the camera world.
        extrinsics, intrinsics = pose_encoding_to_extri_intri(prediction['pose_enc'], tensor.shape[-2:])
        data = {name: prediction[name][0].float().cpu().numpy() for name in ('depth', 'depth_conf')}
        data.update(extrinsic=extrinsics[0].float().cpu().numpy(), intrinsic=intrinsics[0].float().cpu().numpy())
        frames, cameras = {}, []
        for i, (rgb, A, raw_wh) in enumerate(prepared):
            depth = np.asarray(data['depth'][i], np.float32).squeeze()
            K = np.asarray(data['intrinsic'][i], float)
            w2c = np.eye(4); w2c[:3] = np.asarray(data['extrinsic'][i])[:3]
            pose = np.linalg.inv(w2c)
            yy, xx = np.indices(depth.shape)
            rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(K).T
            points = (rays * depth[..., None]) @ pose[:3, :3].T + pose[:3, 3]
            check = (points - pose[:3, 3]) @ pose[:3, :3] @ K.T
            valid = np.isfinite(depth) & (depth > 0)
            residual = np.max(abs(check[valid, :2] / check[valid, 2:] - np.stack([xx, yy], -1)[valid]))
            if not np.isfinite(residual) or residual > 1e-6:
                raise ValueError('Native camera/depth projection round trip failed')
            Image.fromarray(rgb).save(out / f'photo-{i+1}.png')
            with Image.open(paths[i]) as raw:
                source_rgb = np.asarray(raw.convert('RGB'))
            frames[i+1] = dict(K=K, pose=pose, A=A, points=points, depth=depth,
                               canonicalRgb=rgb, rgb=source_rgb)
            np.savez_compressed(out / f'frame-{i+1}.npz', K=K, pose=pose, A=A, depth=depth,
                                points=points.astype(np.float32), rgb=rgb, confidence=np.asarray(data['depth_conf'][i]))
            cameras.append({'photo': i+1, 'K': K.tolist(), 'pose': pose.tolist(), 'A': A.tolist(),
                            'rawWH': list(raw_wh), 'sourceSha256': hashlib.sha256(payloads[i]).hexdigest(),
                            'projectionRoundTripMaxPx': float(residual),
                            'arrayFileOnVolume': str(out / f'frame-{i+1}.npz')})
        save(out / 'cameras.json', {'worldFrame': 'LingBot native', 'frames': cameras})
        export_surface(frames, out)
        fit = fit_and_overlay(frames, frozen, out)
        stats.update(referenceStatus=fit['status'], heldOutPhotos=fit['heldOutPhotos'])
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        save(out / 'failure.json', error)
    stats.update(containerWallSeconds=time.monotonic() - started, codeRevision=REV,
                 weightsRevision=WEIGHTS_REV, weightsSha256=WEIGHTS_SHA,
                 rawPredictionsOnVolume=f'panoptes-lingbot-map:/{run_id}/result/native-prediction.pt',
                 groundTruthUploaded=False, oldGeometryUploaded=False,
                 cameraContract='native pose decoder returns W2C; inverted once to C2W; demo.postprocess unused',
                 configuration={'mode': 'streaming', 'numScaleFrames': 4, 'imageSize': 518,
                                'keyframeInterval': 1, 'cameraIterations': 4, 'seed': 0},
                 comparisonScope='Same four original RGBs, frozen raw contours and dimensions; LingBot replaces MapAnything cameras/depth. Button initialization uses mean camera-up rather than the MapAnything floor normal; fitted axis is free.',
                 nativeUnits=True, error=error)
    save(out / 'run.json', stats); volume.commit()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        for path in sorted(out.iterdir()):
            if path.suffix in ('.json', '.jpg', '.glb', '.png'):
                tar.add(path, arcname=path.name)
    if archive.tell() > 20 * 1024**2:
        raise ValueError('Compact result exceeds 20 MiB; all outputs remain on volume')
    return {'archive': archive.getvalue(), 'run': stats}


@app.local_entrypoint()
def main(images_json: str, baseline: str, measurements: str, out: str):
    paths = [Path(p) for p in json.loads(images_json)]
    if len(paths) != 4 or not all(p.is_file() for p in paths):
        raise ValueError('Supply four original JPEG paths as a JSON array')
    frozen = frozen_inputs(baseline, measurements)
    output = Path(out); output.mkdir(parents=True, exist_ok=False)
    save(output / 'frozen-inputs.json', frozen)
    payloads = [p.read_bytes() for p in paths]
    if len({hashlib.sha256(b).hexdigest() for b in payloads}) != 4:
        raise ValueError('Input photos must be distinct')
    if [hashlib.sha256(b).hexdigest() for b in payloads] != frozen['sourceSha256']:
        raise ValueError('Input order/hashes differ from baseline observations')
    run_id = 'workcell-lingbot-' + str(time.time_ns())
    call_record = {'runId': run_id, 'appId': app.app_id, 'status': 'submitting',
                   'inputSha256': [hashlib.sha256(b).hexdigest() for b in payloads],
                   'adapterSha256': digest(__file__)}
    save(output / 'modal-call.json', call_record)
    started = time.monotonic(); result = None
    try:
        call = probe.spawn(payloads, frozen, run_id)
        call_record.update(callId=call.object_id, status='submitted'); save(output / 'modal-call.json', call_record)
        try:
            result = call.get(timeout=1260)
        except TimeoutError:
            call.cancel(terminate_containers=True)
            raise
        with tarfile.open(fileobj=io.BytesIO(result['archive']), mode='r:gz') as tar:
            for member in tar.getmembers():
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError('Unsafe result archive member')
                (output / member.name).write_bytes(tar.extractfile(member).read())
        call_record['status'] = 'failed' if result['run']['error'] else 'complete'
        if result['run']['error']:
            raise RuntimeError(result['run']['error'])
    except Exception as exc:
        call_record.update(status='failed' if result is not None else 'failed_or_outcome_unknown', error=str(exc))
        raise
    finally:
        elapsed = time.monotonic() - started
        call_record['callWallSeconds'] = elapsed; save(output / 'modal-call.json', call_record)
        save(output / 'spend-ledger.json', {'hardware': 'one ephemeral 2 x A100-80GB container',
             'actualBilledUsd': None, 'appId': app.app_id, 'call': call_record,
             'containerWallSeconds': result['run']['containerWallSeconds'] if result else None,
             'estimatedCostStatus': 'resource time recorded; billing not verified'})
