"""Freeze a new capture into the September (lucida-replica-01) evidence layout, for any list of upright 3:4 photographs.

Same canonical grid, manifest keys, geometry record and object views as prepare_lucida_evidence.py (its object_view is
reused unchanged); the geometry record follows the content rule the platform importer re-checks.

  freeze   --photos image_01.jpg,image_02.jpg --input-dir DIR --output RUN
  geometry --output RUN                  after candidate_pi3x_backend.py (or modal_apps/pi3x_geometry.py) wrote RUN/geometry
  objects  --output RUN --spec SPEC      SPEC: {"objects":[{"object_id","label","reference_frame","views":[{"frame_id","mask","provenance"}]}]}
  floor    --output RUN --spec SPEC      SPEC: {"views":[{"frame_id","mask","provenance"}]}   (masks are original-resolution PNGs)
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_lucida_evidence import digest, object_view, save_json  # noqa: E402

RULE = 'native valid & declared alpha & finite & nonzero points & conf>=0.1'
FRAME_FILES = ('pts3d.npy', 'conf.npy', 'valid_mask.npy', 'content_valid_mask.npy', 'intrinsics.npy', 'camera_to_world.npy', 'canonical.png')


def freeze(input_dir, output, photos):
    output = output.resolve()
    if (output / 'manifest.json').exists():
        raise ValueError('Input manifest is already frozen')
    (output / 'input').mkdir(parents=True, exist_ok=True)
    canonical = output / 'evidence/canonical'
    canonical.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, filename in enumerate(photos, 1):
        origin, frozen = input_dir / filename, output / 'input' / filename
        shutil.copyfile(origin, frozen)
        assert digest(origin) == digest(frozen)
        with Image.open(frozen) as native:
            if native.getexif().get(274, 1) != 1:
                raise ValueError('Requires upright source pixels (EXIF orientation 1)')
            rgb = ImageOps.exif_transpose(native).convert('RGB')
        w, h = rgb.size
        if abs(w / h - .75) > 1e-3:
            raise ValueError('The September canonical grid is portrait 3:4')
        name = f'frame_{index:04d}'
        canvas = Image.new('RGB', (518, 518), 'white')
        canvas.paste(rgb.resize((392, 518), Image.Resampling.LANCZOS), (63, 0))
        canvas.save(canonical / f'{name}.png')
        alpha = np.zeros((518, 518), bool); alpha[:, 63:455] = True
        np.save(canonical / f'{name}_alpha.npy', alpha)
        A = np.array([[392 / w, 0, 63 + (392 / w - 1) / 2], [0, 518 / h, (518 / h - 1) / 2], [0, 0, 1]])
        assert np.allclose((A @ [-.5, -.5, 1])[:2], [62.5, -.5]) and np.allclose((A @ [w - .5, h - .5, 1])[:2], [454.5, 517.5])
        frames.append({'frame_id': name, 'source_upload': str(origin), 'input': str(frozen.relative_to(output)),
                       'sha256': digest(frozen), 'bytes': frozen.stat().st_size, 'width': w, 'height': h,
                       'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                       'canonical': str((canonical / f'{name}.png').relative_to(output)), 'canonical_sha256': digest(canonical / f'{name}.png'),
                       'canonical_shape_hw': [518, 518], 'content_rect_xyxy': [63, 0, 455, 518],
                       'alpha': str((canonical / f'{name}_alpha.npy').relative_to(output)), 'alpha_sha256': digest(canonical / f'{name}_alpha.npy'),
                       'input_to_canonical_pixel_centres': A.tolist(), 'canonical_to_input_pixel_centres': np.linalg.inv(A).tolist(),
                       'resampling': 'Pillow RGB LANCZOS; resize392x518 then white pad x=63..454', 'existing_capture_match': None})
    manifest = {'experiment': output.name, 'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'status': 'inputs_frozen',
                'frames': frames,
                'geometry': {'status': 'pending new joint inference', 'model': 'Pi3X', 'input_frames': [f['frame_id'] for f in frames],
                             'coordinate_system': 'new native OpenCV world', 'metric_scale_known': False},
                'script_sha256': digest(__file__), 'privacy': 'private photographs; geometry and generation on private ephemeral Modal functions'}
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({'output': str(output), 'frames': [f['frame_id'] + ' <- ' + f['input'] for f in frames]}))


def geometry(output):
    """Content masks by the importer's rule and the manifest's geometry record (as written inline for lucida-replica-01)."""
    output = output.resolve(); manifest = json.loads((output / 'manifest.json').read_text())
    meta = json.loads((output / 'geometry/candidate_manifest.json').read_text()); records = []
    for frame in manifest['frames']:
        fid, geom = frame['frame_id'], output / 'geometry/frames' / frame['frame_id']
        frozen = output / frame['canonical']
        if (geom / 'canonical.png').read_bytes() != frozen.read_bytes():
            # The adapter re-encodes the canonical PNG: same pixels required, then one byte-identical copy everywhere.
            assert np.array_equal(np.asarray(Image.open(geom / 'canonical.png').convert('RGB')), np.asarray(Image.open(frozen).convert('RGB')))
            shutil.copyfile(geom / 'canonical.png', frozen); frame['canonical_sha256'] = digest(frozen)
        p = np.load(geom / 'pts3d.npy'); valid = np.load(geom / 'valid_mask.npy').astype(bool); conf = np.load(geom / 'conf.npy')
        alpha = np.load(output / frame['alpha'])
        content = valid & alpha & np.isfinite(p).all(-1) & (np.linalg.norm(p, axis=-1) > 1e-6) & (conf >= .1)
        np.save(geom / 'content_valid_mask.npy', content)
        records.append({'frame_id': fid, 'valid_content_points': int(content.sum()), 'canonical_pixels': int(content.size),
                        'files': {name: digest(geom / name) for name in FRAME_FILES}})
    manifest['geometry'] = {'status': 'complete', 'path': 'geometry', 'model': 'Pi3X', 'input_frames': [f['frame_id'] for f in manifest['frames']],
                            'coordinate_system': 'new native OpenCV world', 'metric_scale_known': False,
                            'inference_seconds': meta.get('inference_and_encoding_seconds'), 'model_load_seconds': meta.get('model_load_seconds'),
                            'frames': records, 'native_pinhole_fit': meta.get('native_pinhole_fit'), 'content_valid_rule': RULE,
                            'device': meta.get('device'), 'precision': meta.get('precision'), 'weights_sha256': meta.get('weights_sha256'),
                            'code_revision': meta.get('code_revision')}
    manifest['status'] = 'geometry_complete'
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({'frames': [{k: r[k] for k in ('frame_id', 'valid_content_points')} for r in records], 'rule': RULE}))


def objects(output, spec_path):
    output = output.resolve(); spec = json.loads(Path(spec_path).read_text()); result = []
    for item in spec['objects']:
        views = [object_view(output, item['object_id'], v['frame_id'], np.asarray(Image.open(v['mask'])) > 0, v['provenance']) for v in item['views']]
        centroids = [np.array(v['centroid_native']) for v in views]
        distances = [float(np.linalg.norm(a - b)) for i, a in enumerate(centroids) for b in centroids[i + 1:]]
        result.append({'object_id': item['object_id'], 'label': item['label'], 'reference_frame': item['reference_frame'],
                       'source_inventory_indices': [], 'views': views,
                       'physical_identity': {'status': item.get('identity', 'single view' if len(views) == 1 else 'same physical object in each listed view (visual review)'),
                                             'evidence': item.get('evidence', 'one photograph only: identity not cross-checked' if len(views) == 1 else
                                                                  f'location and fixed details in all {len(views)} listed photographs (visual review)'),
                                             'pairwise_visible_centroid_distances_native': distances}})
    path = output / 'evidence/objects.json'
    save_json(path, {'version': 1, 'coordinate_system': 'new joint Pi3X native OpenCV world', 'metric_scale_known': False,
                     'status': 'complete for the listed objects', 'objects': result})
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest.setdefault('evidence', {}).update(objects='evidence/objects.json', objects_sha256=digest(path), object_count=len(result),
                                               object_views=sum(len(o['views']) for o in result), script='scripts/research/prepare_capture_evidence.py',
                                               current_script_sha256=digest(__file__))
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({o['object_id']: [round(d, 3) for d in o['physical_identity']['pairwise_visible_centroid_distances_native']] for o in result}))


def floor(output, spec_path):
    """Observed floor plane from floor masks (RANSAC on the joint point maps), as prepare_lucida_evidence.floor_evidence."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ehs_spatial.geometry import _ransac_floor_plane
    output = output.resolve(); spec = json.loads(Path(spec_path).read_text()); manifest = json.loads((output / 'manifest.json').read_text())
    views = [object_view(output, 'observed_floor', v['frame_id'], np.asarray(Image.open(v['mask'])) > 0, v['provenance']) for v in spec['views']]
    points = np.concatenate([np.load(output / v['points_path']) for v in views]); rgb = np.concatenate([np.load(output / v['colors_path']) for v in views])
    cameras = [np.load(output / f'geometry/frames/{f["frame_id"]}/camera_to_world.npy') for f in manifest['frames']]
    up = np.mean([-c[:3, 1] for c in cameras], axis=0); up /= np.linalg.norm(up)
    threshold = .005 * np.linalg.norm(np.quantile(points, .95, axis=0) - np.quantile(points, .05, axis=0))
    fit = points[::max(1, len(points) // 20000)]
    indices = _ransac_floor_plane(fit, np.asarray([c[:3, 3] for c in cameras]), up, threshold)
    if indices is None or len(indices) < 150:
        raise ValueError('Insufficient observed floor plane support')
    center = np.mean(fit[indices], axis=0); normal = np.linalg.svd(fit[indices] - center, full_matrices=False)[2][-1]
    normal *= np.sign(normal @ up); offset = -float(normal @ center)
    residual = np.abs(points @ normal + offset); inside = residual < threshold
    np.save(output / 'evidence/floor_points.npy', points); np.save(output / 'evidence/floor_colors.npy', rgb)
    record = {'plane_native': [float(v) for v in normal] + [offset], 'up_native': normal.tolist(), 'coordinate_system': 'new joint Pi3X native world',
              'metric_scale_known': False, 'fit': 'existing ehs_spatial.geometry._ransac_floor_plane then SVD on its inliers',
              'distance_threshold_native': float(threshold), 'total_observed_points': len(points), 'inlier_points': int(inside.sum()),
              'all_point_residual_median_native': float(np.median(residual)), 'all_point_residual_p95_native': float(np.quantile(residual, .95)),
              'inlier_residual_p95_native': float(np.quantile(residual[inside], .95)),
              'camera_signed_heights_native': [float(c[:3, 3] @ normal + offset) for c in cameras],
              'points_path': 'evidence/floor_points.npy', 'colors_path': 'evidence/floor_colors.npy', 'views': views,
              'evidence_type': 'observed partial geometry only; not a generated asset'}
    assert all(h > 0 for h in record['camera_signed_heights_native'])
    save_json(output / 'evidence/floor.json', record)
    objects_json = json.loads((output / 'evidence/objects.json').read_text()); background, background_rgb = [points], [rgb]
    for obj in objects_json['objects']:
        if 'fence' in obj['object_id']:
            for v in obj['views']:
                background.append(np.load(output / v['points_path'])); background_rgb.append(np.load(output / v['colors_path']))
    np.savez(output / 'evidence/observed_background.npz', xyz=np.concatenate(background), rgb=np.concatenate(background_rgb))
    manifest.setdefault('evidence', {}).update(floor='evidence/floor.json', floor_sha256=digest(output / 'evidence/floor.json'))
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({k: record[k] for k in ('plane_native', 'total_observed_points', 'inlier_points', 'all_point_residual_p95_native', 'camera_signed_heights_native')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'geometry', 'objects', 'floor'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--photos')
    parser.add_argument('--spec', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.input_dir, args.output, args.photos.split(','))
    elif args.mode == 'geometry':
        geometry(args.output)
    elif args.mode == 'objects':
        objects(args.output, args.spec)
    else:
        floor(args.output, args.spec)
