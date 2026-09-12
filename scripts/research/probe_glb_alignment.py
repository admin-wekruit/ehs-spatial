"""Frozen-cache Pi3X GLB -> product MapAnything Sim(3) probe; no inference.

uv run python scripts/research/probe_glb_alignment.py --self-test
uv run python scripts/research/probe_glb_alignment.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = Path('/Users/adam/Desktop/panoptes-public/panoptes-serving')
WORKCELL = PUBLIC / 'outputs/candidate-evaluation/workcell-reconstruction-01'
TARGET = ROOT / 'runs/user-bor1-02'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stats(values):
    a = np.asarray(values)
    return {'count': int(a.size), **{k: float(v) for k, v in zip(
        ['median', 'p90', 'p95', 'max'], np.quantile(a, [.5, .9, .95, 1]))},
        'mean': float(a.mean())} if a.size else {'count': 0}


def apply(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def similarity(x, y, weights=None):
    if len(x) < 3 or x.shape != y.shape:
        raise ValueError('At least three corresponding 3D points required')
    w = np.ones(len(x)) if weights is None else np.asarray(weights)
    w = w / w.sum()
    mx, my = w @ x, w @ y
    xc, yc = x - mx, y - my
    u, singular, vt = np.linalg.svd((yc * w[:, None]).T @ xc)
    parity = np.ones(3)
    parity[-1] = np.linalg.det(u @ vt)
    rotation = (u * parity) @ vt
    variance = np.sum(w[:, None] * xc * xc)
    if variance < 1e-12 or singular[1] < 1e-12:
        raise ValueError('Degenerate point correspondences')
    scale = np.sum(singular * parity) / variance
    if scale <= 0:
        raise ValueError('Nonpositive scale')
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = my - scale * rotation @ mx
    return matrix


def robust_similarity(x, y, threshold=.15):
    rng = np.random.default_rng(20260909)
    # ponytail: bounded 20k training sample and 256 hypotheses; no ICP or nearest-neighbour identity guesses.
    chosen = rng.choice(len(x), min(len(x), 20000), replace=False)
    xx, yy = x[chosen], y[chosen]
    best, best_score = similarity(xx, yy), float('inf')
    for i in range(257):
        try:
            candidate = best if i == 0 else similarity(xx[idx := rng.choice(len(xx), 6, replace=False)], yy[idx])
        except ValueError:
            continue
        score = np.median(np.linalg.norm(apply(candidate, xx) - yy, axis=1))
        if score < best_score:
            best, best_score = candidate, score
    for _ in range(20):
        residual = np.linalg.norm(apply(best, xx) - yy, axis=1)
        weights = np.minimum(1., threshold / np.maximum(residual, 1e-12)) ** 2
        updated = similarity(xx, yy, weights)
        if np.max(np.abs(updated - best)) < 1e-10:
            break
        best = updated
    return best


def residual_report(matrix, x, y):
    error = np.linalg.norm(apply(matrix, x) - y, axis=1)
    return {'distance_target_display_units': stats(error),
            'within': {str(t): float(np.mean(error <= t)) for t in [.05, .10, .15, .25, .50]}}


def load(directory):
    data = {k: np.load(directory / (k + '.npy'), allow_pickle=False).astype(np.float64)
            for k in ['pts3d', 'conf', 'camera_to_world', 'intrinsics']}
    data['valid_mask'] = np.load(directory / 'valid_mask.npy', allow_pickle=False).astype(bool)
    data['image'] = np.array(Image.open(directory / 'canonical.png').convert('RGB'))
    shape = data['valid_mask'].shape
    assert data['pts3d'].shape == (*shape, 3) and data['conf'].shape == shape
    assert data['image'].shape == (*shape, 3)
    c2w = data['camera_to_world']
    assert np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-4)
    assert np.linalg.det(c2w[:3, :3]) > .999
    data['depth'] = ((data['pts3d'] - c2w[:3, 3]) @ c2w[:3, :3])[..., 2]
    data['usable'] = (data['valid_mask'] & np.isfinite(data['pts3d']).all(axis=2)
                      & (np.abs(data['pts3d']).sum(axis=2) > 1e-6) & (data['depth'] > 0))
    return data


def verify_glb_uv(document, binary, json_length, source):
    """Check this frozen single-primitive export directly, bypassing importer UV flips."""
    assert len(document['meshes']) == 1 and len(document['meshes'][0]['primitives']) == 1
    primitive = document['meshes'][0]['primitives'][0]
    assert primitive.get('mode', 4) == 4
    def accessor(index):
        a = document['accessors'][index]
        view = document['bufferViews'][a['bufferView']]
        assert view.get('buffer', 0) == 0 and 'byteStride' not in view and 'sparse' not in a
        width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}[a['type']]
        dtype = {5126: '<f4', 5125: '<u4'}[a['componentType']]
        offset = 28 + json_length + view.get('byteOffset', 0) + a.get('byteOffset', 0)
        return np.frombuffer(binary, dtype=dtype, offset=offset, count=a['count'] * width).reshape(-1, width)
    vertices = accessor(primitive['attributes']['POSITION']).astype(float)
    uv = accessor(primitive['attributes']['TEXCOORD_0']).astype(float)
    indices = accessor(primitive['indices']).ravel()
    assert np.array_equal(indices, np.arange(len(vertices))) and len(vertices) % 3 == 0
    triangles, triangle_uv = vertices.reshape(-1, 3, 3), uv.reshape(-1, 3, 2)
    tiles = np.floor(triangle_uv * 2).astype(int)
    assert np.all(tiles == tiles[:, :1])
    assert np.all(tiles[..., 1] == 1) and set(tiles[..., 0].ravel()) == {0,1}
    frames = []
    for column, number in [(0,3),(1,4)]:
        selected = tiles[:,0,0] == column
        xyz, tex = triangles[selected].reshape(-1,3), triangle_uv[selected].reshape(-1,2)
        frame = load(source / 'frames' / f'frame_{number:04d}')
        c = frame['camera_to_world']
        camera_xyz = (xyz - c[:3,3]) @ c[:3,:3]
        projected = camera_xyz @ frame['intrinsics'].T
        direct = projected[:,:2] / projected[:,2:3]
        decoded = (tex * 2 - [column,1]) * [392,518] + [63,0] - .5
        error = np.linalg.norm(decoded - direct, axis=1)
        assert error.max() < 1e-3
        assert np.all((decoded >= [63,0]) & (decoded <= [454,517]))
        xy = np.floor(direct + .5).astype(int)
        observed_z = frame['depth'][xy[:,1],xy[:,0]]
        relative = np.abs(camera_xyz[:,2] - observed_z) / observed_z
        assert frame['usable'][xy[:,1],xy[:,0]].all()
        assert (frame['conf'][xy[:,1],xy[:,0]] >= .1).all()
        assert relative.max() <= .040001
        frames.append({'frame': f'frame_{number:04d}', 'faces': int(selected.sum()),
                       'uv_vs_native_camera_projection_px': stats(error),
                       'vertex_depth_relative_error': stats(relative)})
    return {'status': 'passed', 'primitive_count': 1, 'face_count': len(triangles),
            'vertex_count': len(vertices), 'indices_sequential': True,
            'all_face_vertices_in_same_atlas_tile': True,
            'raw_glTF_TEXCOORD_0_v_origin': 'top-left; do not flip again',
            'raw_uv_to_frame': 'floor(u*2) + 2*floor(v*2) + 1; actual frames3/4 only',
            'raw_uv_to_canonical': '(uv*2 - [tile_column,tile_row])*[392,518] + [63,0] - 0.5',
            'three_faceIndex': 'Single indexed TRIANGLES primitive and sequential indices: local faceIndex equals frozen exported triangle order. No offset required.',
            'frames': frames}


def main():
    sys.path.insert(0, str(ROOT))
    from ehs_spatial.viewer import _frames
    from ehs_spatial.geometry import _fit_floor
    selection_path = WORKCELL / 'selection-state34-robotcrop.json'
    static_path = WORKCELL / 'selection.json'
    selection, static = (json.loads(p.read_text()) for p in [selection_path, static_path])
    glb_path = WORKCELL / 'pi3x-surface-02/textured-workcell.glb'
    glb_bytes = glb_path.read_bytes()
    magic, version, length, json_length, chunk_type = struct.unpack('<4sIIII', glb_bytes[:20])
    assert (magic, version, length, chunk_type) == (b'glTF', 2, len(glb_bytes), 0x4E4F534A)
    glb = json.loads(glb_bytes[20:20 + json_length])
    for node in glb['nodes']:
        assert np.allclose(np.array(node.get('matrix', np.eye(4).T.ravel())).reshape(4, 4).T, np.eye(4))
        assert node.get('translation', [0,0,0]) == [0,0,0]
        assert node.get('rotation', [0,0,0,1]) == [0,0,0,1]
        assert node.get('scale', [1,1,1]) == [1,1,1]
    texture_metrics = json.loads((glb_path.parent / 'texture-metrics.json').read_text())
    assert sha(glb_path) == texture_metrics['output']['sha256']
    assert texture_metrics['readback']['vertices_match']
    source = Path(selection['source_geometry'])
    uv_contract = verify_glb_uv(glb, glb_bytes, json_length, source)
    target_scale = json.loads((TARGET / 'scene.json').read_text())['scale_factor']
    floor, warnings = _fit_floor({f.frame_id: f for f in _frames(TARGET)}, {}, 1.5,
                                scale_factor_override=target_scale)
    if floor is None:
        raise ValueError(f'Product floor unavailable: {warnings}')
    native_to_floor = np.eye(4)
    native_to_floor[:3, :3] = floor.scale_factor * floor.rotation
    native_to_floor[:3, 3] = -native_to_floor[:3, :3] @ floor.origin
    anchors = json.loads(re.search(r'<script id="anchors"[^>]*>(.*?)</script>',
                                   (TARGET / 'viewer.html').read_text(), re.S).group(1))
    reference = load(source / 'frames' / selection['reference_frame'])
    low, high = np.array(selection['crop_box_camera_min']), np.array(selection['crop_box_camera_max'])
    pairs, records, cameras = {}, [], []
    for number in range(1, 5):
        frame = f'frame_{number:04d}'
        a, b = load(source / 'frames' / frame), load(TARGET / 'geometry/frames' / frame)
        if not np.array_equal(a['image'], b['image']):
            raise ValueError(f'{frame}: canonical RGB differs; no same-pixel matching allowed')
        source_input = sorted((PUBLIC / 'runs/user-bor1-02/input').glob('image_*'))[number - 1]
        target_input = sorted((TARGET / 'input').glob('image_*'))[number - 1]
        assert sha(source_input) == sha(target_input), f'{frame}: original image differs'
        h, w = a['usable'].shape
        yy, xx = np.indices((h, w))
        x0, y0, x1, y1 = selection['content_rect']
        content = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        static_mask = content.copy()
        for x0, y0, x1, y1 in static['exclude_rects'][frame]:
            static_mask[y0:y1, x0:x1] = False
        c = reference['camera_to_world']
        reference_xyz = (a['pts3d'] - c[:3, 3]) @ c[:3, :3]
        crop = ((reference_xyz >= low) & (reference_xyz <= high)).all(axis=2)
        usable = static_mask & crop & a['usable'] & b['usable'] & (a['conf'] >= .1)
        # TSDF input uses fitted K rays and camera-z, not point-map XY; preserve native Pi3X world.
        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
        ray_xyz = (pixels @ np.linalg.inv(a['intrinsics']).T) * a['depth'][..., None]
        pinhole_world = ray_xyz @ a['camera_to_world'][:3, :3].T + a['camera_to_world'][:3, 3]
        tile_holdout = ((xx // 32 + yy // 32) % 2) == 1
        pairs[frame] = {'x': pinhole_world[usable], 'native_x': a['pts3d'][usable],
                        'y': apply(native_to_floor, b['pts3d'][usable]),
                        'xy': pixels[usable, :2],
                        'target_c2w': b['camera_to_world'], 'target_K': b['intrinsics'],
                        'test': tile_holdout[usable]}
        anchor = next(v for v in anchors if v['frame_id'] == frame)
        actual_position = apply(native_to_floor, b['camera_to_world'][:3, 3][None])[0]
        assert np.max(np.abs(actual_position - anchor['position'])) <= 5.1e-5
        cameras.append({'frame': frame, 'source_opencv_c2w': a['camera_to_world'].tolist(),
                        'source_K': a['intrinsics'].tolist(),
                        'target_opencv_c2w': b['camera_to_world'].tolist(),
                        'target_K': b['intrinsics'].tolist(),
                        'target_floor_position': actual_position.tolist(),
                        'exact_input_sha256': sha(source_input)})
        records.append({'frame': frame, 'static_exclusion_rects': static['exclude_rects'][frame],
                        'training_eligible': number in [3, 4],
                        'content_pixels': int(content.sum()), 'static_pixels': int(static_mask.sum()),
                        'static_focus_pixels': int((static_mask & crop).sum()),
                        'usable_correspondences': int(usable.sum()),
                        'source_zero_filled_valid': int((a['valid_mask'] & ~np.any(a['pts3d'] != 0, axis=2)).sum()),
                        'target_zero_filled_valid': int((b['valid_mask'] & ~np.any(b['pts3d'] != 0, axis=2)).sum()),
                        'holdout_pixels': int((tile_holdout & usable).sum()),
                        'native_pointmap_vs_TSDF_rays_source_units': stats(np.linalg.norm(a['pts3d'][usable] - pinhole_world[usable], axis=1)),
                        'source_hashes': {n: sha(source / 'frames' / frame / n) for n in ['pts3d.npy','camera_to_world.npy','intrinsics.npy','canonical.png']},
                        'target_hashes': {n: sha(TARGET / 'geometry/frames' / frame / n) for n in ['pts3d.npy','camera_to_world.npy','intrinsics.npy','canonical.png']}})
    cases = {}
    for name, fit_frames, robust in [('static34_robust', [3,4], True), ('static34_least_squares', [3,4], False),
                                    ('frame3_robust', [3], True), ('frame4_robust', [4], True)]:
        train = [pairs[f'frame_{i:04d}'] for i in fit_frames]
        x = np.concatenate([v['x'][~v['test']] for v in train])
        y = np.concatenate([v['y'][~v['test']] for v in train])
        matrix = robust_similarity(x, y) if robust else similarity(x, y)
        scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
        result = {'source_world_to_target_floor': matrix.tolist(),
                  'source_world_to_target_native': (np.linalg.inv(native_to_floor) @ matrix).tolist(),
                  'scale': scale, 'train': residual_report(matrix, x, y), 'holdout_by_frame': {}}
        for frame, pair in pairs.items():
            sel = pair['test']
            result['holdout_by_frame'][frame] = residual_report(matrix, pair['x'][sel], pair['y'][sel])
            target_native = apply(np.linalg.inv(native_to_floor) @ matrix, pair['x'][sel])
            c = pair['target_c2w']
            camera_xyz = (target_native - c[:3, 3]) @ c[:3, :3]
            projected = camera_xyz @ pair['target_K'].T
            positive = projected[:, 2] > 0
            uv = projected[positive, :2] / projected[positive, 2:3]
            result['holdout_by_frame'][frame]['target_camera_reprojection_px'] = stats(np.linalg.norm(uv - pair['xy'][sel][positive], axis=1))
            result['holdout_by_frame'][frame]['target_positive_depth_fraction'] = float(positive.mean())
        eligible = [pairs[f'frame_{i:04d}'] for i in [3,4]]
        result['holdout34'] = residual_report(matrix, np.concatenate([v['x'][v['test']] for v in eligible]),
                                             np.concatenate([v['y'][v['test']] for v in eligible]))
        result['camera_checks'] = []
        for camera in cameras:
            c2w = np.array(camera['source_opencv_c2w'])
            target_c2w = np.array(camera['target_opencv_c2w'])
            position = apply(matrix, c2w[:3, 3][None])[0]
            source_forward = matrix[:3, :3] @ c2w[:3, 2] / scale
            target_forward = floor.rotation @ target_c2w[:3, 2]
            result['camera_checks'].append({'frame': camera['frame'],
                'position_error_target_units': float(np.linalg.norm(position - camera['target_floor_position'])),
                'forward_error_degrees': float(np.degrees(np.arccos(np.clip(source_forward @ target_forward, -1, 1))))})
        cases[name] = result
    best = cases['static34_robust']
    quality = best['holdout34']
    accepted = (quality['distance_target_display_units']['median'] <= .10
                and quality['distance_target_display_units']['p95'] <= .25
                and quality['within']['0.15'] >= .8)
    result = {'status': 'complete', 'run': 'user-bor1-02', 'source_glb': str(WORKCELL / 'pi3x-surface-02/textured-workcell.glb'),
              'source_glb_sha256': sha(WORKCELL / 'pi3x-surface-02/textured-workcell.glb'),
              'source_glb_nodes_identity_verified': True,
              'source_glb_matches_texture_export_readback_sha': True,
              'source_glb_uv_contract': uv_contract,
              'source_geometry': str(source), 'target_geometry': str(TARGET / 'geometry'),
              'target_native_to_product_floor': native_to_floor.tolist(),
              'target_transform_matches_existing_viewer_camera_anchors': True,
              'metric_accuracy_known': False, 'target_units': 'existing product display units (scene moge_anchor scale); not external ground truth',
              'correspondence': 'Exact same original image SHA + exact canonical RGB + same integer pixel, static rectangles and mesh focus crop. No label or nearest-neighbour matching.',
              'source_positions': 'TSDF camera-z backprojection through original fitted K, in native Pi3X world; exported GLB has no world-coordinate change.',
              'holdout': '32x32 checkerboard image tiles; odd parity reserved before fitting. All holdout residuals retained including outliers. Frame3-only and frame4-only fits also evaluated on other frame.',
              'training': 'Only frames3/4 static focus regions; both source Pi3X conf>=.1, both valid/finite/nonzero/positive-depth. No target confidence threshold.',
              'frame1_caveat': 'Frame1 has no frozen dynamic exclusion rectangle: its diagnostic includes dynamic content and is never fitted.',
              'nominal_display_acceptance': {'median_max': .10, 'p95_max': .25, 'fraction_within_0_15_min': .8,
                  'scope': 'Exploratory display alignment gate, not a physical safety calibration tolerance.'},
              'reliable_single_sim3_for_product_overlay': bool(accepted), 'frames': records, 'cameras': cameras, 'experiments': cases,
              'frame_association': {'reliable_evidence': 'Exact original SHA and canonical pixels identify the same source frame. Render GLB using original Pi3X c2w/K; frames3/4 are the actual mesh evidence.',
                  'canonical_crop': {'xyxy': [63,0,455,518], 'cropped_size_wh': [392,518], 'K_change': 'subtract 63 from cx only; unchanged camera-to-world'},
                  'opencv_to_three_camera': 'camera_world_matrix = source_c2w @ diag(1,-1,-1,1); do not apply candidate Sim3 when showing native mesh',
                  'inventory_link_rule': 'Use exact (frame_id, inv) canonical mask. A triangle can claim support only after projection with that frame Pi3X camera and camera-z agreement to that frame native depth. No label matching.',
                  'frame_1_2': 'Available as original photos/camera references; dynamic robot/load state is different and was not fused. Do not invent corresponding mesh object identities.',
                  'measurement_rule': 'Keep current inventory numbers explicitly from MapAnything; native GLB has unknown physical scale and is not a substitute for those measurements.'},
              'limitations': ['Two learned reconstructions are not ground truth.', 'Static rectangles are conservative manual selections, not semantic segmentation.',
                              'Neighbouring image tiles can see the same surface; leave-frame-out results are included.',
                              'The GLB only fuses frames3/4; frame1/2 robot and load state must not be treated as the same physical snapshot.']}
    out = ROOT / 'outputs/report-glb-integration-20260909'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'alignment-probe.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'alignment-uv-check.json').write_text(json.dumps({'glb_sha256': sha(glb_path), **uv_contract}, indent=2) + '\n')
    print(json.dumps({'accepted': bool(accepted), 'holdout34': quality,
                      'out': str(out / 'alignment-probe.json')}, indent=2))


def self_test():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(1500, 3))
    angle = .7
    exact = np.eye(4)
    exact[:3, :3] = 1.7 * np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    exact[:3, 3] = [.3, -.2, 4.]
    y = apply(exact, x)
    assert np.allclose(similarity(x, y), exact, atol=1e-10)
    y[:300] += rng.normal(0, 5, (300,3))
    fit = robust_similarity(x, y, .02)
    assert np.max(np.abs(fit - exact)) < .001
    assert np.median(np.linalg.norm(apply(fit, x[300:]) - y[300:], axis=1)) < .001
    assert np.linalg.det(fit[:3,:3]) > 0
    print('self-test passed: column transform, positive-scale rotation, 20% outliers, held-clean residual')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else main()
