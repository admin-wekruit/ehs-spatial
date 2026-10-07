"""Public scene and report document of an assembled capture run, for the platform importer (with --geometry-root RUN).

The September equivalents were built in the Pages repository (pack-model.py, build-unified-data.py, build-source-bridge.py)
around a legacy registry; this keeps their scene/report schema and reuses their helpers (floor_transform, pose, contours,
metrics) without the legacy mapping: original photographs, exact camera pixel maps, packed meshes, per-view mask polygons
and plan hulls for the CAD view.

  python scripts/research/build_capture_report.py --run outputs/candidate-evaluation/RUN --label TITLE [--pages-root DIR]
Writes RUN/public/{scene.json, report.json, model/objects/*.bin.gz, images/frame_*.png, originals/*}.
--pages-root (or $PANOPTES_PAGES_ROOT): a directory holding the Pages repository's pack-model.py and build-unified-data.py
(on-prem: ship those two files with the code); default: the local panoptes-workcell-pages checkout.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

PAGES = Path(os.environ.get('PANOPTES_PAGES_ROOT', '/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helpers(pages):
    spec = importlib.util.spec_from_file_location('unified', pages / 'build-unified-data.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def build(run, label, pages=PAGES):
    run = run.resolve(); out = run / 'public'
    missing = [n for n in ('pack-model.py', 'build-unified-data.py') if not (pages / n).is_file()]
    if missing:
        raise FileNotFoundError(f'--pages-root {pages} lacks {missing}')
    if out.exists():
        raise ValueError('RUN/public already exists')
    out.mkdir()
    u = helpers(pages)
    manifest = json.loads((run / 'manifest.json').read_text())
    evidence = {o['object_id']: o for o in json.loads((run / 'evidence/objects.json').read_text())['objects']}
    floor = json.loads((run / 'evidence/floor.json').read_text())
    shutil.copyfile(run / 'result/scene.json', out / 'scene.json')
    subprocess.run([sys.executable, str(pages / 'pack-model.py'), str(run / 'result'), str(out)], check=True)
    scene = json.loads((out / 'scene.json').read_text())
    frames = {f['frame_id']: f for f in manifest['frames']}
    (out / 'images').mkdir(); (out / 'originals').mkdir()
    for camera in scene['cameras']:
        fid = camera['id']; frame = frames[fid]; geom = run / 'geometry/frames' / fid
        K = np.load(geom / 'intrinsics.npy'); C = np.load(geom / 'camera_to_world.npy'); A = np.asarray(frame['input_to_canonical_pixel_centres'])
        assert np.array_equal(K, camera['K']) and np.array_equal(C, camera['camera_to_world']), 'scene camera differs from the frozen geometry'
        shutil.copyfile(run / frame['canonical'], out / 'images' / f'{fid}.png')
        assert (out / 'images' / f'{fid}.png').read_bytes() == (geom / 'canonical.png').read_bytes()
        name = Path(frame['input']).name; shutil.copyfile(run / frame['input'], out / 'originals' / name)
        camera.update(image=f'images/{fid}.png', original_image=f'originals/{name}', original_width=frame['width'], original_height=frame['height'],
                      original_K=(np.linalg.inv(A) @ K.astype(float)).tolist(), input_to_canonical_pixel_centres=A.tolist(), original_sha256=frame['sha256'])
    scene.update(label=label, source_run_id=manifest['experiment'],
                 provenance={**scene.get('provenance', {}), 'source_bridge': {'target_manifest_sha256': sha(run / 'manifest.json'),
                             'note': 'own capture run; no legacy registry'}})
    if manifest.get('evidence', {}).get('floor_sha256'):
        scene['floor_reference'] = {'path': 'evidence/floor.json', 'sha256': manifest['evidence']['floor_sha256']}
    scene.setdefault('floor_plane', floor['plane_native'])
    transform = u.floor_transform(floor)
    objects, plane_points = [], []
    for entry in scene['objects']:
        raw = gzip.decompress((out / entry['mesh']['asset']['path']).read_bytes())
        vertices = np.frombuffer(raw, '<f4', entry['mesh']['vertex_count'] * 9).reshape(-1, 9)
        world_to_floor = transform @ u.pose(entry['transform'])
        points = vertices[:, :3] @ world_to_floor[:3, :3].T + world_to_floor[:3, 3]
        hull = cv2.convexHull(points[:, :2].astype(np.float32)).reshape(-1, 2); moments = cv2.moments(hull)
        plane_points.extend(hull.tolist())
        source_views = evidence[entry['id']]['views'] if entry['id'] in evidence else floor['views'] if entry.get('source') == 'observed' else []
        views = []
        for view in source_views:
            if view['frame_id'] not in entry.get('frame_ids', []):
                continue
            mask = np.load(run / view['canonical_mask_path'], allow_pickle=False).astype(bool)
            polygons, bbox = u.contours(mask)
            views.append({'frame_id': view['frame_id'], 'polygons': polygons, 'bbox': bbox, 'fill_rule': 'evenodd'})
        comparison = run / 'result' / entry['metrics']['comparison'] if entry.get('metrics', {}).get('comparison') else None
        objects.append({'id': entry['id'], 'label': entry['label'], 'source': entry['source'], 'inventory_indices': [], 'views': views,
                        'plan': {'hull': hull.astype(float).round(7).tolist(),
                                 'center': [moments['m10'] / moments['m00'], moments['m01'] / moments['m00']] if moments['m00'] else hull.mean(0).tolist(),
                                 'z_min': float(points[:, 2].min()), 'z_max': float(points[:, 2].max())},
                        'metrics': u.metrics(json.loads(comparison.read_text())) if comparison and comparison.exists() else None})
    cameras = []
    for camera in scene['cameras']:
        c2w = transform @ np.asarray(camera['camera_to_world'], float); forward = c2w[:2, 2]
        cameras.append({'id': camera['id'], 'position': c2w[:2, 3].tolist(), 'forward': (forward / np.linalg.norm(forward)).tolist()})
        plane_points.append(c2w[:2, 3].tolist())
    plane_points = np.asarray(plane_points)
    report = {'source_run_id': manifest['experiment'], 'reconstruction_run_id': scene['run_id'], 'scene_url': 'scene.json',
              'frames': [{'id': c['id'], 'label': c.get('label', c['id']), 'url': c['image'], 'width': c['width'], 'height': c['height']} for c in scene['cameras']],
              'objects': objects,
              'plan': {'bounds': {'min': plane_points.min(0).tolist(), 'max': plane_points.max(0).tolist()}, 'cameras': cameras,
                       'native_to_floor': transform.tolist(), 'unit': 'native relative units'},
              'scale': {'reconstruction': 'uncalibrated'},
              'mapping_notes': ['照片、3D、CAD 投影按相同 scene object id 联动；俯视轮廓是全部网格顶点的凸包，不代表地面接触面。',
                                '每个视图的多边形来自该照片的物体掩码（规范 518 网格）；尺度另行由急停标定写入。']}
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    (out / 'scene.json').write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'objects': [(o['id'], [v['frame_id'] for v in o['views']]) for o in objects], 'cameras': len(cameras),
                      'scene_sha256': sha(out / 'scene.json')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--pages-root', type=Path, default=PAGES, help='directory with pack-model.py and build-unified-data.py')
    args = parser.parse_args()
    build(args.run, args.label, args.pages_root)
