"""Export independently segmented visible RGB-D object surfaces in an existing map.

No completion, back-face generation, ICP, cross-view identity assignment or
training. Each mesh keeps its source pixels and one SAM observation identity.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

from build_replay_scene import media_spans, world_points
from reconstruct_room_rgb import digest, validate_frame
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.providers.sam3 import decode_coco_rle


def observed_surface(color, depth, mask, k, c2w, max_edge_m=.05, *, depth_range=(.2,5)):
    """Triangulate adjacent pixels only; bounds are in the input depth's units."""
    if not 0 <= depth_range[0] < depth_range[1] or not np.isfinite(max_edge_m) or max_edge_m <= 0:
        raise ValueError('Invalid surface depth/edge bounds')
    valid = mask & np.isfinite(depth) & (depth > 0) & (depth >= depth_range[0]) & (depth <= depth_range[1])
    y, x = np.where(valid)
    pixels = np.column_stack((x, y))
    vertices = world_points(pixels, depth[valid], k, c2w)
    indices = np.full(depth.shape, -1, np.int32)
    indices[valid] = np.arange(len(vertices))
    a, b, c, d = indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]
    faces = np.concatenate((np.stack((a, c, b), -1).reshape(-1, 3),
                            np.stack((b, c, d), -1).reshape(-1, 3)))
    faces = faces[(faces >= 0).all(axis=1)]
    if len(faces):
        tri = vertices[faces]
        # A bound in the caller's depth units prevents spanning depth jumps.
        edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
        area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        faces = faces[(edges.max(axis=1) <= max_edge_m) & (area2 > 1e-12)]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, np.int32); remap[used] = np.arange(len(used))
    return vertices[used], remap[faces], color[y[used], x[used]], pixels[used], int(valid.sum())


def read(path):
    return json.loads(Path(path).read_text())


def attach_semantic_review(result, review_dir, output):
    review, manifest = read(review_dir / 'review.json'), read(review_dir / 'input-manifest.json')
    baseline = hashlib.sha256(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if (manifest['source_video_sha256'] != result['source_video_sha256']
            or review['status'] != 'model_interpretation_not_ground_truth'):
        raise ValueError('Semantic review video or interpretation status differs')
    reviewed = {o['id']: o for o in review['observations']}
    expected_ids = {o['entityId'] for o in result['staticObjects']}
    evidence_ids = [o['entityId'] for o in manifest['observations']]
    if (set(reviewed) != expected_ids or len(review['observations']) != len(expected_ids)
            or set(evidence_ids) != expected_ids or len(evidence_ids) != len(expected_ids)):
        raise ValueError('Semantic review observation IDs differ')
    for observation in manifest['observations']:
        if sorted(Path(f['path']).name for f in observation['files']) != ['overlay.png', 'source.png']:
            raise ValueError('Semantic review requires exact source and overlay evidence')
        for source_file in observation['files']:
            target = output / observation['entityId'] / Path(source_file['path']).name
            if digest(target) != source_file['sha256']:
                raise ValueError('Semantic review image/mask input differs')
    for obj in result['staticObjects']:
        item = reviewed[obj['entityId']]
        if item['status'] not in {'clear', 'partial', 'incorrect_prompt'} or not isinstance(item['description'], str):
            raise ValueError('Invalid semantic review result')
        obj['semanticReview'] = {k: item[k] for k in ['status', 'description', 'category']}
    result['semanticReview'] = {'status': review['status'], 'input_scene_sha256': manifest['scene_sha256'],
        'geometry_scene_sha256': baseline, 'binding': 'exact_observation_ids_source_and_overlay_sha256',
        'url': os.path.relpath((review_dir / 'review.json').resolve(), output.resolve()),
        'sha256': digest(review_dir / 'review.json')}


def build(args):
    import trimesh

    started = time.monotonic()
    scene, native = read(args.scene), read(args.native_scene)
    if (scene['schema'] != 'phase2-replay-scene-v1' or scene['units'] != 'meters'
            or scene['provenance']['native_scene_sha256'] != digest(args.native_scene)
            or native['evaluation_alignment_applied']
            or (scene['coordinate_frame'], scene['map_id']) != (native['coordinate_frame'], native['map_id'])):
        raise ValueError('Expected the same native metric map, without GT alignment')
    run = Path(native['source_run']); meta = read(run / 'run.json')
    if not meta['execution_complete'] or digest(run / 'run.json') != native['source_run_sha256']:
        raise ValueError('Camera run changed or is incomplete')
    for name in ['input.manifest.json', 'camera.yaml']:
        if digest(run / name) != meta['artifacts_sha256'][name]:
            raise ValueError('Camera source changed: ' + name)
    if not scene['provenance'].get('sensor_depth') or meta['sensor'] != 'rgbd':
        raise ValueError('Object surfaces require declared real sensor depth')
    k = np.asarray(scene['provenance']['k'], float)
    factor = scene['provenance']['depth_factor']
    settings = cv2.FileStorage(str(run / 'camera.yaml'), cv2.FILE_STORAGE_READ)
    try:
        calibration = np.array([[settings.getNode('Camera1.fx').real(), 0, settings.getNode('Camera1.cx').real()],
                                [0, settings.getNode('Camera1.fy').real(), settings.getNode('Camera1.cy').real()], [0, 0, 1]])
        if not np.array_equal(k, calibration) or factor != settings.getNode('RGBD.DepthMapFactor').real():
            raise ValueError('Replay calibration differs from camera input')
    finally:
        settings.release()
    inputs = {f['source_index']: f for f in read(run / 'input.manifest.json')}
    native_frames = {f['source_index']: f for f in native['frames']}
    frames = {f['sourceFrame']: f for f in scene['frames']}
    if scene.get('staticObjects'):
        raise ValueError('Use the original replay scene; do not overwrite prior object observations')
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    video = Path(read(args.discovery[0] / 'input-manifest.json')['source_clip'])
    if digest(video) != scene['source_video_sha256']:
        raise ValueError('Discovery video differs from the replay video')
    spans, shape = media_spans(video)
    result = copy.deepcopy(scene)
    result['staticObjects'] = []
    result['provenance']['object_model_source_scene_sha256'] = digest(args.scene)
    result['limitations'].append('可选对象仅是独立关键帧的可见表面；不补背面，不合并跨视角身份，也不证明其他时刻仍在原位。')
    if scene.get('meshUrl'):
        result['meshUrl'] = os.path.relpath((args.scene.parent / scene['meshUrl']).resolve(), args.output.resolve())
    args.output.mkdir(parents=True)
    reports = []
    for folder in args.discovery:
        source = read(folder / 'input-manifest.json'); index = source['source_frame_index']
        if (source['prompt'] not in {'chair', 'table', 'monitor'}
                or source['source_clip_sha256'] != scene['source_video_sha256']
                or index not in frames or index not in inputs
                or (source['height'], source['width']) != shape
                or abs(source['timestamp_seconds'] - spans[index][0]) > .001):
            raise ValueError('Discovery source frame, time, category or pixel domain differs')
        frame, camera_input = frames[index], inputs[index]
        if digest(native_frames[index]['source_image']) != camera_input['sha256']:
            raise ValueError('Camera source image changed')
        c2w = np.asarray(frame['c2w'], float)
        if not np.array_equal(c2w, native_frames[index]['c2w']):
            raise ValueError('Source observation camera changed from native map')
        image_path = folder / f'frame-{index}.png'
        if digest(image_path) != source['frame_sha256']:
            raise ValueError('SAM input image changed')
        cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        try:
            ok, decoded = cap.read()
        finally:
            cap.release()
        image = cv2.imread(str(image_path))
        if not ok or not np.array_equal(decoded, image):
            raise ValueError('SAM image is not the exact decoded source frame')
        depth_path = Path(camera_input['depth_source_path'])
        if (digest(depth_path) != camera_input['depth_sha256']
                or abs(camera_input['depth_timestamp'] - native_frames[index]['timestamp']) >= .02):
            raise ValueError('Depth changed or is not associated with this RGB exposure')
        sensor = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if sensor.dtype != np.uint16 or sensor.shape != shape:
            raise ValueError('Expected registered source-domain uint16 sensor depth')
        depth = sensor.astype(np.float32) / factor
        color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        validate_frame(color, depth, np.ones(shape), depth > 0, k, c2w)
        provider = read(folder / 'provider-output.json')
        submissions = [json.loads(line)['data']['request_id'] for line in (folder / 'provider-events.jsonl').read_text().splitlines()
                       if line.startswith('{') and json.loads(line).get('phase') == 'submitted']
        for instance in read(folder / 'instances.json'):
            ordinal = instance['instance_index']
            mask_path = folder / f'instance-{ordinal}-mask.png'
            if digest(mask_path) != instance['mask_sha256']:
                raise ValueError('SAM instance mask changed')
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            if mask.shape != shape or int(mask.sum()) != instance['mask_area_pixels']:
                raise ValueError('SAM mask pixel domain/area differs')
            if not np.array_equal(mask, decode_coco_rle(provider['rle'][ordinal], height=shape[0], width=shape[1]).astype(bool)):
                raise ValueError('Saved mask differs from the original provider RLE')
            entity_id = f"obs-{source['source_clip_sha256'][:12]}-f{index}-{source['prompt']}-{ordinal}"
            model_dir = args.output / entity_id; model_dir.mkdir()
            shutil.copyfile(image_path, model_dir / 'source.png')
            shutil.copyfile(mask_path, model_dir / 'mask.png')
            rgba = np.zeros((*shape, 4), np.uint8); rgba[mask] = [246, 161, 69, 150]
            if not cv2.imwrite(str(model_dir / 'overlay.png'), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
                raise IOError('Could not save source mask overlay')
            vertices, faces, colors, pixels, depth_pixels = observed_surface(color, depth, mask, k, c2w, args.max_edge_m)
            report = {'entityId': entity_id, 'sourceFrame': index, 'label': source['prompt'],
                      'maskPixels': int(mask.sum()), 'validDepthPixels': depth_pixels,
                      'vertices': len(vertices), 'triangles': len(faces),
                      'status': 'observed_surface' if len(faces) else 'no_supported_surface',
                      'source': source, 'instance': instance, 'depth_path': str(depth_path),
                      'discovery_directory': str(folder.resolve()), 'provider_request_ids': submissions,
                      'provider_score': (provider.get('scores') or [None] * len(provider['rle']))[ordinal],
                      'label_status': 'model_prediction_not_manually_confirmed',
                      'depth_sha256': digest(depth_path), 'k': k.tolist(), 'c2w': c2w.tolist(),
                      'native_scene_sha256': digest(args.native_scene),
                      'provider_output_sha256': digest(folder / 'provider-output.json'),
                      'max_edge_m': args.max_edge_m, 'back_faces_completed': False,
                      'cross_view_identity': 'not_assigned'}
            if len(faces):
                model = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
                model.export(model_dir / 'surface.glb')
                reread = trimesh.load(model_dir / 'surface.glb', force='mesh', process=False)
                if not np.array_equal(reread.faces, faces) or not np.allclose(reread.vertices, vertices, atol=1e-6, rtol=0):
                    raise ValueError('GLB changed measured vertex order or faces')
                if not np.array_equal(reread.visual.vertex_colors[:, :3], colors):
                    raise ValueError('GLB changed source vertex colors')
                np.savez_compressed(model_dir / 'source-pixels.npz', xy=pixels,
                                    source_frame=np.full(len(vertices), index, np.int32))
                local = (reread.vertices - c2w[:3, 3]) @ c2w[:3, :3]
                uvw = local @ k.T
                error = np.max(np.abs(uvw[:, :2] / uvw[:, 2:] - pixels))
                if error > .001:
                    raise ValueError('Exported surface does not reproject to original mask pixels')
                report.update({'mesh_sha256': digest(model_dir / 'surface.glb'), 'max_reprojection_error_px': float(error)})
                rel = model_dir.name
                result['staticObjects'].append({'entityId': entity_id, 'label': source['prompt'],
                    'displayName': f"{source['prompt']} · 观测 {ordinal + 1}", 'meshUrl': f'{rel}/surface.glb',
                    'representation': 'single_frame_observed_surface', 'identityScope': 'independent_observation',
                    'source': {'sourceFrame': index, 'timeSec': frame['timeSec'], 'endTimeSec': frame['endTimeSec'],
                               'width': shape[1], 'height': shape[0], 'maskUrl': f'{rel}/overlay.png',
                               'imageUrl': f'{rel}/source.png', 'bbox': instance['mask_bounds_xyxy_exclusive']},
                    'provenanceUrl': f'{rel}/provenance.json'})
            (model_dir / 'provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            reports.append(report)
    if args.semantic_review:
        attach_semantic_review(result, args.semantic_review, args.output)
    (args.output / 'scene.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    summary = {'elapsedSeconds': time.monotonic() - started, 'objects': len(result['staticObjects']),
               'observations': len(reports), 'triangles': sum(r['triangles'] for r in reports),
               'scene_sha256': digest(args.output / 'scene.json'), 'cross_view_fusion': False}
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def self_check():
    depth = np.ones((4, 6), np.float32); depth[:, 3:] = 2
    mask = np.ones(depth.shape, bool); mask[1, 1] = False
    color = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    k = np.diag([100., 100., 1]); c2w = np.eye(4); c2w[:3, 3] = [3, 4, 5]
    v, f, c, uv, _ = observed_surface(color, depth, mask, k, c2w)
    assert len(f) and mask[uv[:, 1], uv[:, 0]].all()
    assert np.array_equal(c, color[uv[:, 1], uv[:, 0]])
    assert np.allclose(v, world_points(uv, depth[uv[:, 1], uv[:, 0]], k, c2w))
    assert (np.ptp(v[f, 2], axis=1) == 0).all(), 'Do not bridge a depth discontinuity'
    assert not len(observed_surface(color, depth, np.zeros_like(mask), k, c2w)[1])
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp:
        output = Path(temp); folder = output / 'obs-check'; folder.mkdir()
        files = []
        for name in ['source.png', 'overlay.png']:
            path = folder / name; path.write_bytes(name.encode())
            files.append({'path': str(path), 'sha256': digest(path)})
        manifest = {'scene_sha256': 'a' * 64, 'source_video_sha256': 'b' * 64,
                    'observations': [{'entityId': 'obs-check', 'files': files}]}
        review = {'status': 'model_interpretation_not_ground_truth',
                  'observations': [{'id': 'obs-check', 'status': 'clear', 'description': 'test evidence', 'category': 'chair'}]}
        (output / 'input-manifest.json').write_text(json.dumps(manifest))
        (output / 'review.json').write_text(json.dumps(review))
        scene = {'source_video_sha256': 'b' * 64, 'staticObjects': [{'entityId': 'obs-check'}]}
        attach_semantic_review(scene, output, output)
        assert scene['semanticReview']['input_scene_sha256'] == 'a' * 64
        assert scene['semanticReview']['geometry_scene_sha256'] != 'a' * 64
        (folder / 'overlay.png').write_bytes(b'changed mask')
        try:
            attach_semantic_review(scene, output, output)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed visual evidence must invalidate semantic reuse')
    print('PASS: exact measured XYZ/color/pixel correspondence, mask holes, no depth bridges/back faces, semantic reuse across geometry and rejection of changed evidence')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path)
    p.add_argument('--native-scene', type=Path)
    p.add_argument('--discovery', type=Path, nargs='+')
    p.add_argument('--output', type=Path)
    p.add_argument('--max-edge-m', type=float, default=.05)
    p.add_argument('--semantic-review', type=Path, help='Existing bounded photo review directory; validates exact observation IDs and source/overlay hashes, permitting camera-only reconstruction')
    p.add_argument('--self-check', action='store_true')
    args = p.parse_args()
    if args.self_check:
        self_check()
    elif not (args.scene and args.native_scene and args.discovery and args.output) or not 0 < args.max_edge_m <= .1:
        p.error('Provide scene, native-scene, discovery directories and a new output; max-edge-m must be in (0,.1]')
    else:
        build(args)
