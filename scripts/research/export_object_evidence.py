"""Export a product run's generic instance evidence for the existing generator.

No models, cloud calls, physical-identity guesses, or writes to the source run.
Only exact payloads passing the official erosion/depth preflight enter objects.json.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

from generate_lucida_assets import payload_for_object, camera_depth, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export(source, output, ehs_repo, candidate_id=None):
    sys.path.insert(0, str(ehs_repo.resolve()))
    from ehs_spatial.object_evidence import build_object_evidence, load_candidate_mask
    from ehs_spatial.providers.map_anything import input_mask_to_canonical
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a new output directory outside the source run')
    registry = build_object_evidence(source)
    if candidate_id is not None and sum(c['id'] == candidate_id for c in registry['candidates']) != 1:
        raise ValueError('Unknown or duplicate requested candidate ID')
    source_manifest_path = source / 'manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.is_file() else {}
    source_manifest_sha = digest(source_manifest_path) if source_manifest else None
    output.mkdir(parents=True)
    # Files are independent copies: downstream writes cannot mutate source inodes.
    for relative, sha in registry['source_sha256'].items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
        assert digest(destination) == sha
    frames, data = [], {}
    for frame in registry['frames']:
        fid = frame['frame_id']
        spec = {'frame_id': fid, 'input': frame['image_path'], 'sha256': frame['source_sha256'],
                'width': frame['width'], 'height': frame['height'],
                'canonical_shape_hw': [frame['canonical_height'], frame['canonical_width']],
                'content_rect_xyxy': frame.get('content_rect_xyxy'),
                'input_mask_transform': frame.get('input_mask_transform'),
                'input_to_canonical_pixel_centres': frame['input_to_canonical_pixel_centres']}
        frames.append(spec)
        directory = output / 'geometry/frames' / fid
        try:
            points = np.load(directory / 'pts3d.npy', allow_pickle=False)
            valid = np.load(directory / 'valid_mask.npy', allow_pickle=False).astype(bool)
            C = np.load(directory / 'camera_to_world.npy', allow_pickle=False)
            _, keep = camera_depth(points, C, valid)
            if frame.get('content_rect_xyxy'):
                x0, y0, x1, y1 = frame['content_rect_xyxy']
                content = np.zeros(valid.shape, bool); content[y0:y1, x0:x1] = True
                keep &= content
            np.save(directory / 'content_valid_mask.npy', keep)
            data[fid] = (frame, points, keep)
        except (OSError, ValueError):
            # Such instances remain blocked in the complete registry below.
            continue
    write(output / 'manifest.json', {'experiment': output.name, 'source_run_id': source.name,
        'frames': frames, 'geometry_coordinate_space': 'unchanged native source-run world',
        'geometry': {'model': (source_manifest.get('geometry') or {}).get('model') or
                              (source_manifest.get('providers') or {}).get('mapanything_model_id'),
                     'metric_scale_known': False, 'source_manifest_sha256': source_manifest_sha},
        'source_floor': {'path': 'scene.json', 'sha256': registry['source_sha256'].get('scene.json'),
                         'coordinate_space': 'unchanged native source-run world'},
        'source_files_sha256': registry['source_sha256'], 'new_model_calls': 0})
    objects, rejected, not_selected = [], [], []
    for candidate in registry['candidates']:
        oid = candidate['id']
        if candidate_id is not None and oid != candidate_id:
            not_selected.append(oid)
            continue
        if candidate['generation']['status'] == 'blocked':
            rejected.append({'object_id': oid, 'reason': candidate['generation']['reason']})
            continue
        try:
            fid = candidate['frame_id']
            frame, points, keep = data[fid]
            mask = load_candidate_mask(source, candidate)
            canonical = input_mask_to_canonical(mask, source, fid, keep.shape)
            directory = output / 'evidence/objects' / oid / fid
            directory.mkdir(parents=True)
            Image.fromarray(mask.astype(np.uint8)*255).save(directory / 'mask.png')
            np.save(directory / 'canonical_mask.npy', canonical)
            np.save(directory / 'points.npy', points[keep & canonical])
            rgb = np.asarray(Image.open(output / frame['canonical_image_path']).convert('RGB'))
            np.save(directory / 'colors.npy', rgb[keep & canonical])
            geom = output / 'geometry/frames' / fid
            relative = lambda p: p.relative_to(output).as_posix()
            view = {'frame_id': fid, 'rgb_path': frame['image_path'], 'mask_path': relative(directory / 'mask.png'),
                    'canonical_mask_path': relative(directory / 'canonical_mask.npy'),
                    'points_path': relative(directory / 'points.npy'), 'colors_path': relative(directory / 'colors.npy'),
                    'canonical_rgb_path': relative(geom / 'canonical.png'), 'pointmap_path': relative(geom / 'pts3d.npy'),
                    'content_valid_path': relative(geom / 'content_valid_mask.npy'), 'valid_path': relative(geom / 'valid_mask.npy'),
                    'conf_path': relative(geom / 'conf.npy'), 'K_path': relative(geom / 'intrinsics.npy'),
                    'c2w_path': relative(geom / 'camera_to_world.npy'),
                    'sha256': {p.name: digest(p) for p in directory.iterdir()},
                    'provenance': {'source_run_id': source.name, 'candidate_id': oid,
                                   'mask_resolution': candidate['mask']['resolution'], 'source_refs': candidate['source_refs']}}
            obj = {'object_id': oid, 'label': candidate['label'], 'reference_frame': fid, 'views': [view],
                   'generation_input': candidate['generation']['input_mode'],
                   'source_inventory_indices': candidate['inventory_indices'],
                   'physical_identity': 'single-frame evidence; cross-view identity is not established',
                   'source_run_id': source.name}
            payload, record = payload_for_object(output, obj)
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                eroded = binary_erosion(arrays['0_mask'] > 0, structure=np.ones((5, 5), bool))
                supported = eroded & (arrays['0_depth'] > 0)
                y, x = np.nonzero(supported)
                if obj['generation_input'] == 'original_rgb_crop':
                    left, top, _, _ = record['source_views'][0]['source_crop_xyxy']
                    pixels = np.stack([x+left, y+top, np.ones_like(x)], -1)
                    canonical_pixels = pixels @ np.asarray(frame['input_to_canonical_pixel_centres']).T
                    cx = np.floor(canonical_pixels[:, 0]+.5).astype(int)
                    cy = np.floor(canonical_pixels[:, 1]+.5).astype(int)
                    independent = len(np.unique(cy * keep.shape[1] + cx))
                else:
                    independent = len(x)
                if independent < 8:
                    raise ValueError('Fewer than 8 independent canonical depth pixels survive official 5x5 erosion')
            candidate['generation'] = {'status': 'ready', 'input_mode': obj['generation_input'], 'reason': None,
                'preflight': {'payload_sha256': hashlib.sha256(payload).hexdigest(),
                              'supported_input_pixels': int(supported.sum()), 'independent_depth_pixels': independent,
                              'mask_erosion': {'kernel_size': 5, 'iterations': 1}}}
            obj['preflight'] = candidate['generation']['preflight']
            objects.append(obj)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            reason = str(error).replace(str(source), '<source run>').replace(str(output), '<derived run>')
            candidate['generation'] = {**candidate['generation'], 'status': 'blocked', 'reason': reason}
            rejected.append({'object_id': oid, 'reason': reason})
    write(output / 'evidence/objects.json', {'source_run_id': source.name, 'objects': objects, 'unavailable_objects': rejected,
                                          'not_selected': not_selected})
    registry['derived_run_id'] = output.name
    write(output / 'object-evidence.json', registry)
    assert all(digest(source / name) == sha for name, sha in registry['source_sha256'].items()), 'Source run changed during export'
    assert source_manifest_sha is None or digest(source_manifest_path) == source_manifest_sha, 'Source manifest changed'
    assert len(objects) + len(rejected) + len(not_selected) == len(registry['candidates'])
    assert len({o['object_id'] for o in objects}) == len(objects)
    validation = {'status': 'passed', 'source_run_id': source.name, 'derived_run_id': output.name,
                  'total_candidates': len(registry['candidates']), 'payload_ready': len(objects), 'blocked': len(rejected),
                  'not_selected':len(not_selected), 'requested_candidate_id':candidate_id,
                  'new_model_calls': 0, 'new_geometry_inference': False,
                  'cross_view_identity': 'not inferred; one source frame per candidate',
                  'objects_sha256': digest(output / 'evidence/objects.json')}
    write(output / 'export-validation.json', validation)
    print(json.dumps(validation, ensure_ascii=False), flush=True)
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ehs-repo', required=True, type=Path)
    parser.add_argument('--candidate-id')
    args = parser.parse_args()
    export(args.source_run, args.output, args.ehs_repo, args.candidate_id)
