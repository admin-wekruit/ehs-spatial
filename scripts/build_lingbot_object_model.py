"""Reuse the pinned photo RecGen transport for one video object observation.

Prepare locally first; --invoke explicitly submits one bounded, journaled call.
Generated unseen surfaces remain estimates, with no new identity association.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.platform.recgen import RecGenRequest, adapt_output, source_grid_crop
from build_lingbot_replay import read_prediction, resize_mask, source_transform
from reconstruct_room_rgb import digest


def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def prepare(scene_path, run, entity):
    scene = read(scene_path)
    if scene['coordinate_frame'] != 'lingbot_native_monocular':
        raise ValueError('Expected native LingBot scene')
    obj = next(o for o in scene['staticObjects'] if o['entityId'] == entity)
    proof = read(scene_path.parent / obj['provenanceUrl'])
    source = obj['source']; index = source['sourceFrame']
    execution = read(run / 'run.json')
    if digest(run / 'run.json') != scene['provenance']['execution_sha256']:
        raise ValueError('Native run differs from scene')
    prediction = next(f for f in execution['frames'] if f['sourceFrame'] == index)
    native_path = run / prediction['file']
    image_path, mask_path = (scene_path.parent / source[k] for k in ['imageUrl', 'maskUrl'])
    for path, expected in [(image_path, proof['source_image_sha256']), (mask_path, proof['source_mask_sha256']),
                           (native_path, proof['native_prediction_sha256'])]:
        if digest(path) != expected: raise ValueError('Input evidence changed: '+str(path))
    if prediction['sha256'] != proof['native_prediction_sha256'] or proof['entityId'] != entity or proof['sourceFrame'] != index:
        raise ValueError('Observation/prediction binding differs')
    rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), -1)[:, :, 3] > 0
    if rgb.shape[:2] != (source['height'], source['width']) or mask.shape != rgb.shape[:2]:
        raise ValueError('Source pixel domain differs')
    depth, confidence, _, k, c2w, valid = read_prediction(native_path)
    frame = next(f for f in scene['frames'] if f['sourceFrame'] == index)
    if (not np.array_equal(c2w, frame['c2w']) or source['timeSec'] != frame['timeSec']
            or not source['timeSec'] < source['endTimeSec'] <= frame['endTimeSec']):
        raise ValueError('Source time/camera differs from replay')
    mapping, _, _ = source_transform(mask.shape)
    crop = source_grid_crop(rgb, mask, np.where(valid & (confidence >= 1.5), depth, 0).astype('float32'), k, mapping)
    payload = {'entityId': entity, 'anchorObservationId': entity, 'seed': 42, 'views': [{
        'observationId': entity, 'observationRevision': 1, 'imageId': scene['source_video_sha256']+f':{index}',
        'imageSha256': digest(image_path), 'maskSha256': digest(mask_path),
        'geometrySolutionSha256': digest(native_path), 'coordinateFrameId': scene['coordinate_frame'],
        'cameraToWorld': c2w, **crop}]}
    request = RecGenRequest.from_payload(payload)
    source_record = {**obj, 'scene_path': str(scene_path.resolve()), 'scene_sha256': digest(scene_path),
        'native_run': str(run.resolve()), 'source_video_sha256': scene['source_video_sha256'],
        'pixel_mapping': crop['pixelMapping'], 'camera_to_world': c2w.tolist(),
        'input_sha256': hashlib.sha256(request.to_npz()).hexdigest(),
        'view_source_hashes': {key: payload['views'][0][key] for key in ['imageSha256', 'maskSha256', 'geometrySolutionSha256']}}
    return payload, source_record, (depth, confidence, valid, resize_mask(mask, mask.shape), k, c2w)


def evaluate(vertices, faces, depth, confidence, valid, mask, k, c2w):
    import open3d as o3d
    camera = (vertices-c2w[:3, 3]) @ c2w[:3, :3]
    cast = o3d.t.geometry.RaycastingScene()
    cast.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(camera.astype('float32')),
        o3d.core.Tensor(faces.astype('uint32'))))
    h, w = mask.shape
    z = cast.cast_rays(cast.create_rays_pinhole(k, np.eye(4), w, h))['t_hit'].numpy()
    visible = np.isfinite(z) & (z > 0)
    overlap = visible & mask & valid & (confidence >= 1.5)
    errors = abs(z[overlap]-depth[overlap])/depth[overlap]
    report = {'supported_pixels': int(overlap.sum()), 'silhouette_iou': float((visible & mask).sum()/max(1, (visible | mask).sum())),
        'relative_depth_median': float(np.median(errors)) if len(errors) else None,
        'relative_depth_p95': float(np.percentile(errors, 95)) if len(errors) else None,
        'metric_scale_validated': False, 'unseen_surfaces_validated': False,
        'comparison': 'source segmentation and estimated monocular depth, not field accuracy',
        'gate': {'min_supported_pixels': 300, 'min_iou': .65, 'max_depth_median': .04, 'max_depth_p95': .10}}
    report['accepted_source_consistency'] = bool(len(errors) >= 300 and report['silhouette_iou'] >= .65
        and report['relative_depth_median'] <= .04 and report['relative_depth_p95'] <= .10)
    return report


def attach(scene_path, output, entity):
    report = read(output / 'validation.json')
    if not report['accepted_source_consistency'] or report['mesh_sha256'] != digest(output / 'model.glb'):
        raise ValueError('Generated model did not pass source consistency')
    scene = read(scene_path)
    def relocate(value):
        if isinstance(value, list):
            for item in value: relocate(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key.endswith('Url') and isinstance(item, str) and not item.startswith(('https://', 'http://')):
                    value[key] = os.path.relpath((scene_path.parent / item).resolve(), output.resolve())
                else: relocate(item)
    relocate(scene)
    obj = next(o for o in scene['staticObjects'] if o['entityId'] == entity)
    obj['generatedModel'] = {'meshUrl': 'model.glb', 'sha256': report['mesh_sha256'],
        'provenanceUrl': 'validation.json', 'sourceFrame': obj['source']['sourceFrame'],
        'status': 'source_consistent_model_estimate'}
    scene['limitations'].append(f"{obj['label']} 可切换预训练生成模型：可见部分经原图检查，遮挡面为推测，未验证后续位置或实际尺寸。")
    target = output / 'scene.json'
    if target.exists() and read(target) != scene: raise ValueError('Immutable scene changed')
    save(target, scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['scene', 'run', 'output']: parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--function-id', required=True)
    parser.add_argument('--invoke', action='store_true')
    parser.add_argument('--attach', action='store_true', help='Attach an already checked model, without a new provider call')
    args = parser.parse_args()
    payload, source, geometry = prepare(args.scene, args.run, args.entity)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'input-manifest.json'
    if manifest_path.exists() and read(manifest_path) != source: raise ValueError('Immutable run input changed')
    save(manifest_path, source)
    (args.output / 'input.npz').write_bytes(RecGenRequest.from_payload(payload).to_npz())
    print(json.dumps({'status': 'prepared', 'shape': list(payload['views'][0]['depth'].shape), 'source': source['input_sha256']}), flush=True)
    if not args.invoke:
        if args.attach: attach(args.scene, args.output, args.entity)
        return
    from ehs_spatial.platform.recgen_transport import invoke
    import trimesh
    os.environ['PANOPTES_RECGEN_JOURNAL'] = str(args.output / 'journal')
    response = invoke(payload, {'modalApp': 'lucida-private-assets', 'modalFunction': 'generate_object',
        'modalFunctionId': args.function_id, 'modalVolume': 'panoptes-lucida-weights'})
    save(args.output / 'provider-record.json', {k:v for k,v in response.items() if k not in ['vertices', 'faces', 'colors', 'officialPosedVertices']})
    if response.get('providerError'): raise ValueError(response['providerError'])
    result = adapt_output(RecGenRequest.from_payload(payload), response)
    mesh = trimesh.Trimesh(result['vertices'], result['faces'], vertex_colors=result['colors'], process=False)
    mesh.apply_transform(result['proposedObjectToNative'])
    mesh.export(args.output / 'model.glb')
    # Validate the serialized deliverable, independently of the adapter arrays.
    exported = trimesh.load(args.output / 'model.glb', force='mesh', process=False)
    report = evaluate(exported.vertices, exported.faces, *geometry)
    save(args.output / 'validation.json', {**report, 'mesh_sha256': digest(args.output / 'model.glb'),
        'provenance': result['provenance'], 'object_to_native': result['proposedObjectToNative'].tolist()})
    print(json.dumps(report), flush=True)
    if args.attach: attach(args.scene, args.output, args.entity)


if __name__ == '__main__': main()
