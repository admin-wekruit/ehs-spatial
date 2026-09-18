"""Run bounded pretrained human-mesh keyframes and align them to the RGB-D map.

Provider results stay immutable. Native MHR topology enables explicitly marked
short-gap vertex interpolation; estimates that fail mask/depth checks are kept
in the audit but never enter replay. No training or ground-truth camera input.
"""
from __future__ import annotations
import argparse, base64, concurrent.futures, hashlib, json, os, shutil, sys, time, urllib.request
from pathlib import Path
import cv2
import numpy as np
from reconstruct_room_rgb import digest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_apps.sam3_video_fal import execute


def save_inputs(folder, image, mask, source):
    """Validate resumed inputs before overwriting any immutable source evidence."""
    encoded = {}
    for name, pixels in [('image', image), ('mask', mask.astype('uint8') * 255)]:
        ok, data = cv2.imencode('.png', pixels)
        if not ok: raise ValueError('Could not encode body input')
        encoded[name] = data.tobytes()
    manifest = {**source, **{name + '_sha256': hashlib.sha256(data).hexdigest() for name, data in encoded.items()}}
    path = folder / 'input-manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest: raise ValueError('Resumed body input differs')
    for name, data in encoded.items(): (folder / (name + '.png')).write_bytes(data)
    path.write_text(json.dumps(manifest, indent=2))


def check_interpolations(scene, base, analysis_path, inputs):
    """Check every displayed in-between mesh against that frame, without refitting."""
    import trimesh
    import open3d as o3d
    analysis = {f['sourceFrame']: f for f in json.loads(analysis_path.read_text())['frames']}
    frames = {f['sourceFrame']: f for f in scene['frames']}
    keys = scene['bodyKeyframes']; meshes = {}
    for b in keys:
        path = base / b['meshUrl']
        if digest(path) != b['mesh_sha256']: raise ValueError('Body mesh changed')
        meshes[b['meshUrl']] = trimesh.load(path, force='mesh', process=False)
    k = np.array(scene['provenance']['k']); records = []
    for entity_id in sorted({b['entityId'] for b in keys}):
        ordered = sorted((b for b in keys if b['entityId'] == entity_id), key=lambda b: b['sourceFrame'])
        for a, b in zip(ordered, ordered[1:]):
            if b['timeSec'] - a['timeSec'] > .38 or a['topology_sha256'] != b['topology_sha256']: continue
            if any(i not in frames or not any(o['entityId'] == entity_id for o in frames[i]['objects']) for i in range(a['sourceFrame'], b['sourceFrame'] + 1)): continue
            ma, mb = meshes[a['meshUrl']], meshes[b['meshUrl']]
            if not np.array_equal(ma.faces, mb.faces): raise ValueError('Body topology differs')
            for index in range(a['sourceFrame'] + 1, b['sourceFrame']):
                f = frames[index]; c = np.array(f['c2w']); t = (f['timeSec'] - a['timeSec']) / (b['timeSec'] - a['timeSec'])
                vertices = ((ma.vertices * (1-t) + mb.vertices * t) - c[:3, 3]) @ c[:3, :3]
                obj = next(o for o in analysis[index]['objects'] if o['entityId'] == entity_id)
                mask = cv2.imread(str(analysis_path.parent / obj['maskUrl']), -1)[:, :, 3] > 0
                path = Path(inputs[index]['depth_source_path'])
                if digest(path) != inputs[index]['depth_sha256']: raise ValueError('Depth changed')
                depth = cv2.imread(str(path), -1) / scene['provenance']['depth_factor']
                cast = o3d.t.geometry.RaycastingScene()
                cast.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices.astype('float32')), o3d.core.Tensor(ma.faces.astype('uint32'))))
                z = cast.cast_rays(cast.create_rays_pinhole(k, np.eye(4), mask.shape[1], mask.shape[0]))['t_hit'].numpy()
                visible = np.isfinite(z); overlap = visible & mask & (depth > .2) & (depth < 5)
                residual = abs(z[overlap] - depth[overlap]); iou = float((visible & mask).sum() / max(1, (visible | mask).sum()))
                median = float(np.median(residual)) if len(residual) else None
                p95 = float(np.percentile(residual, 95)) if len(residual) else None
                accepted = len(residual) >= 500 and iou >= .65 and median <= .08 and p95 <= .2
                records.append({'sourceFrame': index, 'entityId': entity_id, 'keyframes': [a['sourceFrame'], b['sourceFrame']],
                    'status': 'accepted_model_estimate' if accepted else 'rejected_alignment', 'silhouette_iou': iou,
                    'depth_median_m': median, 'depth_p95_m': p95, 'depth_support_pixels': len(residual)})
    scene['bodyInterpolation'] = records
    return records


def align(folder, frame, source, k, factor):
    import trimesh
    import open3d as o3d
    output = json.loads((folder / 'provider-output.json').read_text())
    people = output['metadata']['people']
    if len(people) != 1 or len(output['meshes']) != 1: raise ValueError('Expected one mask-conditioned person')
    person = people[0]; mesh_path = folder / 'provider-mesh.ply'
    if not mesh_path.exists():
        item = output['meshes'][0]; urllib.request.urlretrieve(item['url'] if isinstance(item, dict) else item, mesh_path)
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    joints, pixels = np.array(person['keypoints_3d'], float), np.array(person['keypoints_2d'], float)
    if joints.shape != (70, 3) or pixels.shape != (70, 2) or not np.isfinite(joints).all() or not np.isfinite(pixels).all():
        raise ValueError('Invalid anatomical keypoints')
    ok, rotation, translation = cv2.solvePnP(joints, pixels, k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok: raise ValueError('Camera calibration solve failed')
    uv, _ = cv2.projectPoints(joints, rotation, translation, k, None)
    error = np.linalg.norm(uv[:, 0] - pixels, axis=1)
    # Fal PLY is OpenGL camera space including pred_cam_t; MHR keypoints are root-relative OpenCV.
    vertices = mesh.vertices * [1, -1, -1] - np.array(person['pred_cam_t'])
    vertices = vertices @ cv2.Rodrigues(rotation)[0].T + translation.ravel()
    mask = cv2.imread(str(folder / 'mask.png'), 0) > 0
    depth_path = Path(source['depth_source_path'])
    if digest(depth_path) != source['depth_sha256']: raise ValueError('Depth source changed')
    depth = cv2.imread(str(depth_path), -1) / factor
    triangle = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices.astype('float32')), o3d.core.Tensor(mesh.faces.astype('uint32')))
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(triangle)
    rays = scene.create_rays_pinhole(k, np.eye(4), mask.shape[1], mask.shape[0])
    predicted = scene.cast_rays(rays)['t_hit'].numpy()
    overlap = np.isfinite(predicted) & mask & (depth > .2) & (depth < 5)
    if overlap.sum() < 500: raise ValueError('Insufficient visible depth for anatomical alignment')
    metric_scale = np.median(depth[overlap] / predicted[overlap])
    residual = abs(depth[overlap] - predicted[overlap] * metric_scale)
    iou = (np.isfinite(predicted) & mask).sum() / (np.isfinite(predicted) | mask).sum()
    report = {'sourceFrame': frame['sourceFrame'], 'representation': 'inferred_anatomical_mesh',
              'source_image_sha256': digest(folder / 'image.png'), 'source_mask_sha256': digest(folder / 'mask.png'),
              'provider_output_sha256': digest(folder / 'provider-output.json'), 'provider_mesh_sha256': digest(mesh_path),
              'depth_sha256': source['depth_sha256'], 'camera': frame['c2w'], 'k': k.tolist(),
              'pnp_error_px_p95': float(np.percentile(error, 95)), 'silhouette_iou': float(iou),
              'depth_error_m_median': float(np.median(residual)), 'depth_error_m_p95': float(np.percentile(residual, 95)),
              'depth_support_pixels': int(overlap.sum()), 'depth_scale': float(metric_scale),
              'quality_gate': {'min_iou': .65, 'max_depth_median_m': .08, 'max_depth_p95_m': .2, 'max_pnp_p95_px': 5},
              'field_measurement_validated': False}
    accepted = iou >= .65 and np.median(residual) <= .08 and np.percentile(residual, 95) <= .2 and np.percentile(error, 95) <= 5 and .5 <= metric_scale <= 2
    report['status'] = 'accepted_model_estimate' if accepted else 'rejected_alignment'
    if accepted:
        c = np.array(frame['c2w']); mesh.vertices = vertices * metric_scale @ c[:3, :3].T + c[:3, 3]
        mesh.visual.vertex_colors = [178, 207, 222, 255]
        mesh.export(folder / 'body.glb'); report['mesh_sha256'] = digest(folder / 'body.glb')
        report['topology_sha256'] = __import__('hashlib').sha256(np.asarray(mesh.faces, dtype=np.uint32).tobytes()).hexdigest()
        report['vertices'] = len(mesh.vertices)
    (folder / 'alignment.json').write_text(json.dumps(report, indent=2))
    return report


def run(args):
    started = time.monotonic()
    source = json.loads(args.scene.read_text()); analysis = json.loads(args.analysis.read_text())
    native = json.loads(args.native_scene.read_text()); run = Path(native['source_run'])
    if (source['provenance']['native_scene_sha256'] != digest(args.native_scene)
            or source['provenance']['analysis_sha256'] != digest(args.analysis)
            or native['evaluation_alignment_applied'] or not source['provenance']['sensor_depth']
            or source['source_video_sha256'] != digest(args.video)):
        raise ValueError('Inconsistent camera/mask/video source')
    inputs = {f['source_index']: f for f in json.loads((run / 'input.manifest.json').read_text())}
    af = {f['sourceFrame']: f for f in analysis['frames']}; frames = {f['sourceFrame']: f for f in source['frames']}
    k = np.array(source['provenance']['k']); factor = source['provenance']['depth_factor']
    candidates = []
    for frame in source['frames']:
        index = frame['sourceFrame']
        if index % args.stride: continue
        for obj in af.get(index, {}).get('objects', []):
            if any(o['entityId'] == obj['entityId'] and o.get('surface') for o in frame['objects']):
                candidates.append((frame, obj))
    if args.resume:
        plan=json.loads((args.output / 'plan.json').read_text())
        if plan['source_scene_sha256']!=digest(args.scene) or plan['calls']!=len(candidates):raise ValueError('Resume source changed')
    else: args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_text(json.dumps({'source_scene_sha256': digest(args.scene), 'calls': len(candidates),
        'maximum_usd': args.max_usd, 'stride_frames': args.stride,
        'endpoint': 'fal-ai/sam-3/3d-body', 'training': False}, indent=2))
    cap = cv2.VideoCapture(str(args.video)); jobs = []
    try:
        for frame, obj in candidates:
            index = frame['sourceFrame']; track = obj['entityId'].rsplit('-', 1)[-1]
            folder = args.output / f'frame-{index:05d}-track-{track}'; folder.mkdir(exist_ok=args.resume)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index); ok, image = cap.read(); assert ok
            rgba = cv2.imread(str(args.analysis.parent / obj['maskUrl']), -1); mask = rgba[:, :, 3] > 0
            save_inputs(folder, image, mask, {'sourceFrame': index, 'entityId': obj['entityId'],
                'source_video_sha256': source['source_video_sha256']})
            if args.reuse_run and not (folder/'provider-output.json').exists():
                cached=args.reuse_run/folder.name
                if (cached/'provider-output.json').exists():
                    if json.loads((cached/'input-manifest.json').read_text()) != json.loads((folder/'input-manifest.json').read_text()): raise ValueError('Cached body input differs')
                    for name in ['provider-output.json','provider-mesh.ply']:
                        if (cached/name).exists(): shutil.copyfile(cached/name,folder/name)
                    (folder/'reused-from.json').write_text(json.dumps({'source':str(cached.resolve()),'provider_output_sha256':digest(cached/'provider-output.json'),'new_provider_submission':False},indent=2))
            jobs.append((folder, frame, obj))
    finally: cap.release()
    new_jobs=sum(not (folder/'reused-from.json').exists() for folder,_,_ in jobs)
    if new_jobs > args.max_calls or args.max_calls * .02 > args.max_usd: raise ValueError('New keyframes exceed the explicitly bounded budget')
    plan=json.loads((args.output/'plan.json').read_text());plan.update({'new_provider_calls':new_jobs,'reused_provider_results':len(jobs)-new_jobs,'published_estimate_usd':new_jobs*.02})
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    if args.prepare_only: print(json.dumps(plan),flush=True);return
    quotes=[]
    for events in args.output.glob('*/provider-events*.jsonl'):
        for event in map(json.loads,events.read_text().splitlines()):
            if event['phase']=='pricing':quotes.append({'pricing':event['data'],'fetched_at':events.stat().st_mtime})
    quote=max(quotes,key=lambda q:q['fetched_at']) if quotes else None
    if quote and time.time()-quote['fetched_at']>500: quote=None
    def process(job):
        folder, frame, obj = job
        try:
            if not (folder/'provider-output.json').exists():
                events=[json.loads(line) for path in folder.glob('*.jsonl') for line in path.read_text().splitlines()]
                submissions={e['data']['request_id']:e['data'] for e in events if e['phase']=='submitted'}
                if len(submissions)>1:raise ValueError('Ambiguous previous submissions')
                payload={'mode': 'recover' if submissions else 'submit', 'endpoint': 'fal-ai/sam-3/3d-body',
                         'billing_units': 1, 'max_fal_usd': .02,
                         'input': {'image_url': 'data:image/png;base64,' + base64.b64encode((folder / 'image.png').read_bytes()).decode(),
                                   'mask_url': 'data:image/png;base64,' + base64.b64encode((folder / 'mask.png').read_bytes()).decode(),
                                   'export_meshes': True, 'include_3d_keypoints': True, 'include_mhr_params': True}}
                if submissions:payload['submission']=next(iter(submissions.values()))
                if quote:payload['batch_pricing_quote']=quote
                execute(payload,folder,f'provider-events-{time.time_ns()}.jsonl')
            report = align(folder, frame, inputs[frame['sourceFrame']], k, factor)
            return {**report, 'entityId': obj['entityId'], 'folder': folder.name}
        except Exception as e:
            report = {'status': 'failed', 'error_type': type(e).__name__, 'message':str(e)[:160], 'sourceFrame': frame['sourceFrame'], 'entityId': obj['entityId'], 'folder': folder.name}
            (folder / 'failure.json').write_text(json.dumps(report)); return report
    # Obtain one current account price before the batch; repeated pricing GETs
    # previously hit rate limits even though the inference budget was valid.
    first=next((job for job in jobs if not (job[0]/'provider-output.json').exists()),None)
    reports=[]
    if first:
        reports.append(process(first))
        if not (first[0]/'provider-output.json').exists(): raise RuntimeError('First request incomplete; recover its saved request ID before continuing')
        for path in sorted(first[0].glob('provider-events*.jsonl')):
            for event in map(json.loads,path.read_text().splitlines()):
                if event['phase']=='pricing': quote={'pricing':event['data'],'fetched_at':path.stat().st_mtime}
    # Three independently bounded requests; no resubmit on timeout or ambiguous completion.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: reports += list(pool.map(process, [job for job in jobs if job is not first]))
    reports.sort(key=lambda r:(r['sourceFrame'],r['entityId']))
    result = json.loads(args.scene.read_text())
    def relocate(url): return os.path.relpath((args.scene.parent / url).resolve(), args.output.resolve())
    for key in ['meshUrl', 'pointCloudUrl']:
        if result.get(key): result[key] = relocate(result[key])
    for obj in result.get('staticObjects', []):
        for key in ['meshUrl', 'provenanceUrl']: obj[key] = relocate(obj[key])
        for key in ['maskUrl', 'imageUrl']: obj['source'][key] = relocate(obj['source'][key])
    if result.get('semanticReview'): result['semanticReview']['url'] = relocate(result['semanticReview']['url'])
    for frame in result['frames']:
        for obj in frame['objects']:
            if obj.get('surface'): obj['surface']['meshUrl'] = relocate(obj['surface']['meshUrl'])
    result['bodyKeyframes'] = [{**r, 'meshUrl': r['folder'] + '/body.glb', 'timeSec': frames[r['sourceFrame']]['timeSec']} for r in reports if r['status'] == 'accepted_model_estimate']
    result['bodyModelPolicy'] = {'method': 'SAM 3D Body; calibrated PnP and sensor-depth scale', 'maxInterpolationSeconds': .38,
        'interpolation': 'linear_vertex_same_track_same_topology_only_when_observed', 'hidden_surfaces': 'model_inference_not_sensor_observation'}
    check_interpolations(result, args.output, args.analysis, inputs)
    result['limitations'].append('完整人体是预训练模型估计；仅在同轨迹相邻有效关键帧间插值，最长0.38秒，遮挡部分不代表实测。')
    (args.output / 'scene.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    (args.output / 'reports.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps({'elapsedSeconds': time.monotonic() - started, 'requested': len(jobs), 'accepted': len(result['bodyKeyframes']), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['scene', 'native-scene', 'analysis', 'video', 'output']: p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--reuse-run',type=Path,help='Reuse only byte-identical source image/mask provider results')
    p.add_argument('--stride',type=int,default=10,choices=range(1,31))
    p.add_argument('--max-calls', type=int, required=True); p.add_argument('--max-usd', type=float, required=True)
    run(p.parse_args())
