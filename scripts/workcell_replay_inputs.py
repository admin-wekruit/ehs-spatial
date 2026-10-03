"""Rebuild the frozen 2026-10-02 workcell run directory from public artifacts.

No GPU and no Mac-only cache: the public release package
``workcell-photo-handoff-2026-09-30`` carries the MapAnything frames of the
same capture, and the public report directory carries every later model,
floor and catalog file. The 2026-10-01 run reused the identical network output
(cameras are bit-identical) and only replaced the depth-gradient mask with the
resolution-normalized rule of ``be93511``. That mask is replayed here and every
valid point is compared bit for bit with the published capture point clouds;
any mismatch aborts. Files copied from the report must match the SHA-256 that
the frozen G experiment recorded for its input; files without such a record are
listed separately with their observed hashes. Nothing is fitted or edited.

python scripts/workcell_replay_inputs.py --release EXTRACTED_PACKAGE \
  --pages panoptes-workcell-report/workcell-photo-direct --out NEW_DIR
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from modal_apps.mapanything_app import _encode_array, _relative_depth_gradient  # noqa: E402

BASELINE = 'research-notes/workcell-three-boards-complete-2026-09-30-a'
# Report files whose exact bytes the frozen G experiment hashed as its input.
FROZEN = ('geometry.json', 'objects.json', 'posts.glb', 'fence-fitted.glb', 'floor-fitted.glb',
          'physical-clearances.json', 'model-endpoint-estimate.json', 'measurements.json',
          'cart-single.glb', 'guard-left.glb', 'guard-center.glb', 'guard-right.glb',
          'object-extras.glb', 'robot-v1.glb', 'robot-v2.glb', 'robot-v3.glb', 'robot-v4.glb',
          'photo-1.png', 'photo-2.png', 'photo-3.png', 'photo-4.png')
# Packaging inputs of the same published report without an independent record.
PUBLISHED = ('robot-multi.glb', 'guard-multi.glb', 'guard-partition.json', 'guard-placement.json',
             'cart-observed.glb', 'fence-observed.glb', 'structural-result.json',
             'guard-left-initializer.glb', 'guard-right-initializer.glb',
             'mask-contact-sheet.jpg', 'extra-mask-contact-sheet.jpg', 'cart-mask-sheet.jpg',
             'geometry-anchor.jpg', *[f'geometry-evidence-{i}.jpg' for i in range(1, 5)],
             *[f'structural-overlay-{i}.jpg' for i in range(1, 5)],
             *[f'raw-image-features-{i}.jpg' for i in range(1, 5)])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array(spec):
    return np.frombuffer(base64.b64decode(spec['data']), np.dtype(spec['dtype'])).reshape(spec['shape'])


def _glb_points(path):
    data = Path(path).read_bytes()
    length = struct.unpack('<I', data[12:16])[0]
    head = json.loads(data[20:20 + length])
    count = head['accessors'][0]['count']
    blob = data[28 + length:]
    return np.frombuffer(blob[:12 * count], '<f4').reshape(count, 3)


def replay_frame(frame):
    """Apply the 2026-10-01 mask rule to a 2026-09-30 frame of the same inference."""
    points, pose = _array(frame['pts3d']), _array(frame['camera_poses']).astype(np.float64)
    # Camera-frame depth of the stored points; the network depth_z itself was not retained.
    depth = ((points.astype(np.float64) - pose[:3, 3]) @ pose[:3, :3])[..., 2].astype(np.float32)
    mask = _array(frame['non_ambiguous_mask']).astype(bool) & (_relative_depth_gradient(depth, frame) < .08)
    return {**frame, 'non_ambiguous_mask': _encode_array(mask), 'model_revision': None}


def replay(release, pages, out):
    release, pages, out = Path(release), Path(pages), Path(out)
    if out.exists():
        raise ValueError('Output must be a new directory')
    manifest = json.loads((pages / 'housing-boundaries/input-manifest.json').read_text())['sha256']
    report = json.loads((pages / 'scene-report.json').read_text())
    transform = np.asarray(report['sceneTransformNative'], float)
    sources = {}
    for i in range(1, 5):
        path = release / 'inputs' / f'image_{i:02d}.jpg'
        if _sha(path) != manifest[f'source-{i}.jpg']:
            raise ValueError(f'Original photo {i} differs from the frozen input')
        sources[i] = path
    out.mkdir(parents=True)
    record = {'schemaVersion': 1, 'revisionId': report['revision']['id'],
              'method': 'MapAnything 2026-09-30 frames + 2026-10-01 resolution-normalized depth-gradient mask; report files by frozen hash',
              'frames': {}, 'frozen': {}, 'published': {}, 'release': {}, 'absent': []}
    baseline = release / BASELINE
    for i in range(1, 5):
        frame = json.loads(gzip.decompress((baseline / f'frame_{i:04d}.json.gz').read_bytes()))
        replayed = replay_frame(frame)
        points = _array(replayed['pts3d'])
        valid = _array(replayed['non_ambiguous_mask']) & np.isfinite(points).all(-1)
        expected = _glb_points(pages / f'entity-points-capture-photo-{i}.glb')
        actual = np.ascontiguousarray(trimesh.transform_points(points[valid], transform), '<f4')
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f'Photo {i}: replayed valid points differ from the published capture cloud')
        name = f'frame_{i:04d}.json.gz'
        with gzip.GzipFile(out / name, 'wb', mtime=0) as stream:
            stream.write(json.dumps(replayed).encode())
        record['frames'][name] = {'sha256': _sha(out / name), 'validPixels': int(valid.sum()),
                                  'sourceSha256': _sha(baseline / name), 'originalSha256': manifest[name],
                                  'originalShaRecordedBy': 'housing-boundaries/input-manifest.json',
                                  'publishedPointsSha256': _sha(pages / f'entity-points-capture-photo-{i}.glb'),
                                  'check': 'transformed valid points equal the published capture cloud bit for bit',
                                  'omitted': ['depth_z (not retained by the 2026-09-30 frames; no workcell consumer)']}
    for name in FROZEN:
        if _sha(pages / name) != manifest[name]:
            raise ValueError(f'{name}: published bytes differ from the frozen input')
        shutil.copyfile(pages / name, out / name)
        record['frozen'][name] = manifest[name]
    if _sha(baseline / 'sam3.json') != manifest['sam3.json']:
        raise ValueError('sam3.json differs from the frozen input')
    shutil.copyfile(baseline / 'sam3.json', out / 'sam3.json')
    record['release']['sam3.json'] = manifest['sam3.json']
    measured = json.loads((out / 'measurements.json').read_text())
    (out / 'reference-input.json').write_text(json.dumps(measured['reference']))
    if _sha(out / 'reference-input.json') != manifest['reference-input.json']:
        raise ValueError('reference-input.json does not reproduce the frozen input')
    record['frozen']['reference-input.json'] = manifest['reference-input.json']
    for name in PUBLISHED:
        if (pages / name).is_file():
            shutil.copyfile(pages / name, out / name)
            record['published'][name] = _sha(out / name)
        else:
            record['absent'].append(name)
    # The published semantic experiment and its crops; binding re-verifies its inputs.
    semantic = pages / 'semantic'
    if semantic.is_dir():
        shutil.copytree(semantic, out / 'semantic-experiment')
        record['published']['semantic-experiment'] = {path.relative_to(semantic).as_posix(): _sha(path)
                                                      for path in sorted(semantic.rglob('*')) if path.is_file()}
    # Camera provenance from the run's own packaged page, never from the replayed frames.
    data = json.loads((pages / 'data.json').read_text())
    summaries = []
    for i, camera in enumerate(data['frames'], 1):
        frame = json.loads(gzip.decompress((out / f'frame_{i:04d}.json.gz').read_bytes()))
        if (not np.array_equal(_array(frame['intrinsics']), np.asarray(camera['K'], np.float32)) or
                not np.array_equal(_array(frame['camera_poses']), np.asarray(camera['cameraToWorld'], np.float32))):
            raise ValueError(f'Photo {i}: replayed camera differs from the published run cameras')
        summaries.append({'photo': i, 'sourceSha256': _sha(sources[i]),
                          'inputToCanonicalPixelCentres': frame['input_mask_transform']['input_to_canonical_pixel_centres'],
                          'K': camera['K'], 'cameraPose': camera['cameraToWorld'],
                          'validPixels': record['frames'][f'frame_{i:04d}.json.gz']['validPixels']})
    (out / 'geometry-timing.json').write_text(json.dumps({
        'frameSummaries': summaries, 'replayed': True,
        'source': 'cameras from published data.json of the same run; timings of the original inference are not part of this replay'}, indent=2) + '\n')
    record['geometryTiming'] = 'frameSummaries rebuilt from published run cameras; original wall/peak-memory timings unavailable'
    # The original run's own recorded timing and model checks; this replay repeats no inference.
    timing = data['timing']
    record['originalMetrics'] = {'oneShotWallSeconds': timing['oneShotSeconds'], 'robotQuality': data['quality'],
                                 'cartQuality': data['cartQuality'],
                                 'stageTiming': {key: timing[key] for key in ('geometrySeconds', 'segmentationSeconds',
                                                                               'owlSeconds', 'cartMaskSeconds', 'modelSeconds')},
                                 'source': 'published data.json of revision ' + record['revisionId']}
    (out / 'replay-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--release', type=Path, required=True, help='Extracted workcell-photo-reproduction-2026-09-30 package root')
    parser.add_argument('--pages', type=Path, required=True, help='Published workcell-photo-direct directory')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.release, args.pages, args.out)
    print(json.dumps({'output': str(args.out), 'frames': len(result['frames']), 'frozen': len(result['frozen']),
                      'published': len(result['published']), 'absent': result['absent']}))
