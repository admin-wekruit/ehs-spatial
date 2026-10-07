"""Fit two explicit cylinder priors to frozen workcell evidence, without inference.

Run .venv/bin/python scripts/research/fit_blender_posts.py --root EXPERIMENT.
The candidate never changes the generated scene. --self-check checks the saved fit.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares, minimize
import trimesh

from assemble_lucida_scene import digest, load_view, score_view, write_json


IDS = ('left_post', 'right_post')
DECISION_GATE = {'rule': 'mean IoU >= current generated_refined AND mean relative_depth_p50 <= current generated_refined',
                 'status': 'Specified after the initial candidate scores were available; exploratory decision gate, not predeclared validation',
                 'target_masks': 'Full original evidence masks retained, including observed base plates',
                 'promotes_source_scene': False}


def decision_gate(comparisons):
    for comparison in comparisons:
        means = {side: {metric: float(np.mean([v[side][metric] for v in comparison['views']]))
                        for metric in ('visible_iou', 'relative_depth_p50')}
                 for side in ('before_generated_refined', 'after_cylinder')}
        before, after = means['before_generated_refined'], means['after_cylinder']
        comparison['available_view_mean'] = means
        comparison['meets_exploratory_decision_gate'] = (after['visible_iou'] >= before['visible_iou'] and
                                                        after['relative_depth_p50'] <= before['relative_depth_p50'])


def floor_basis(floor):
    plane = np.asarray(floor['plane_native'], dtype=float)
    plane /= np.linalg.norm(plane[:3])
    normal = plane[:3]
    x = np.array([1., 0., 0.])
    if abs(x @ normal) > .9:
        x = np.array([0., 1., 0.])
    x -= (x @ normal) * normal
    x /= np.linalg.norm(x)
    rotation = np.column_stack([x, np.cross(normal, x), normal])
    return rotation, -plane[3] * normal


def pose(parameters, rotation, origin):
    result = np.eye(4)
    result[:3, :3] = rotation @ np.diag([parameters['radius_native']] * 2 + [parameters['height_native']])
    result[:3, 3] = origin + rotation @ np.asarray(parameters['center_floor_native'])
    return result


def seed_from_points(points, rotation, origin):
    local = (points - origin) @ rotation
    lo, hi = np.quantile(local[:, 2], [.02, .98])
    height = float(hi - lo)
    assert height > 0 and np.isfinite(local).all()
    # ponytail: middle-height cross sections exclude the cap/base plate; a
    # cylinder cannot represent those parts and is not an equipment CAD model.
    body = local[(local[:, 2] > lo + .2 * height) & (local[:, 2] < lo + .8 * height), :2]
    assert len(body) >= 30, 'Too little observed post body'
    center = np.median(body, axis=0)
    span = float(np.linalg.norm(np.quantile(body, .95, axis=0) - np.quantile(body, .05, axis=0)))
    assert span > 0
    radius = span / 2
    estimate = np.linalg.lstsq(np.column_stack([2 * body, np.ones(len(body))]), (body * body).sum(1), rcond=None)[0]
    radius_squared = estimate[2] + estimate[:2] @ estimate[:2]
    if radius_squared > 0 and .1 * span < np.sqrt(radius_squared) < 2 * span:
        center, radius = estimate[:2], float(np.sqrt(radius_squared))
    fit = least_squares(lambda p: np.linalg.norm(body - p[:2], axis=1) - p[2],
                        [*center, radius], loss='soft_l1', f_scale=.1 * radius,
                        bounds=([center[0] - span, center[1] - span, .05 * span],
                                [center[0] + span, center[1] + span, 2 * span]), max_nfev=100)
    return {'center_floor_native': [float(fit.x[0]), float(fit.x[1]), float((lo + hi) / 2)],
            'radius_native': float(fit.x[2]), 'height_native': height}, {
        'observed_point_count': len(points), 'body_point_count': len(body),
        'observed_z_quantiles_02_98_native': [float(lo), float(hi)],
        'circle_radial_residual_p50_native': float(np.median(np.abs(fit.fun))),
        'circle_fit_success': bool(fit.success), 'circle_fit_evaluations': fit.nfev}


def check(root):
    output = root / 'blender-01/post-fit'
    record = json.loads((output / 'parameters.json').read_text())
    comparisons = json.loads((output / 'comparisons.json').read_text())
    assert comparisons['decision_gate'] == DECISION_GATE
    assert record['vlm_calls'] == 0 and record['new_model_inference_calls'] == 0
    assert record['metric_scale_known'] is False and record['promoted_to_source_scene'] is False
    assert [o['object_id'] for o in record['objects']] == list(IDS)
    for obj, comparison in zip(record['objects'], comparisons['objects']):
        assert obj['object_id'] == comparison['object_id']
        params = obj['fitted_parameters']
        assert params['radius_native'] > 0 and params['height_native'] > 0
        matrix = np.asarray(obj['native_object_to_world'])
        assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
        assert np.allclose(matrix, pose(params, np.asarray(record['floor_frame_to_native_rotation']),
                                        np.asarray(record['floor_frame_origin_native'])))
        assert np.allclose(matrix[3], [0, 0, 0, 1]) and np.linalg.det(matrix[:3, :3]) > 0
        assert np.allclose(np.linalg.norm(matrix[:3, :3], axis=0),
                           [params['radius_native']] * 2 + [params['height_native']])
        assert np.allclose(matrix[:3, 2] / params['height_native'], record['floor_normal_native'])
        assert comparison['views'] and len(comparison['views']) == len(obj['frame_ids'])
        for view in comparison['views']:
            for name in ('before_generated_refined', 'after_cylinder'):
                metrics = view[name]
                assert 0 <= metrics['visible_iou'] <= 1
                assert metrics['target_pixels'] > 0 and metrics['depth_comparison_pixels'] > 0
                assert all(v is None or np.isfinite(v) for v in metrics.values())
            assert view['before_generated_refined']['target_pixels'] == view['after_cylinder']['target_pixels']
        independently_scored = {'views': comparison['views']}
        decision_gate([independently_scored])
        assert independently_scored['meets_exploratory_decision_gate'] == comparison['meets_exploratory_decision_gate']
    result = {'status': 'passed', 'checked_objects': list(IDS),
              'parameters_sha256': digest(output / 'parameters.json'),
              'comparisons_sha256': digest(output / 'comparisons.json')}
    write_json(output / 'self-check.json', result)
    return result


def run(root):
    started = time.perf_counter()
    output = root / 'blender-01/post-fit'
    output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((root / 'evidence/objects.json').read_text())
    floor = json.loads((root / 'evidence/floor.json').read_text())
    rotation, origin = floor_basis(floor)
    cylinder = trimesh.creation.cylinder(radius=1, height=1, sections=64)
    objects, comparisons = [], []
    for object_id in IDS:
        obj_started = time.perf_counter()
        source = next(o for o in evidence['objects'] if o['object_id'] == object_id)
        views = [load_view(root, spec, 144) for spec in source['views']]
        points = np.concatenate([view['points'][view['mask'] & view['valid']] for view in views])
        initial, seed_data = seed_from_points(points, rotation, origin)
        initial_center = np.asarray(initial['center_floor_native'])
        radius, height = initial['radius_native'], initial['height_native']
        def candidate(delta):
            return {'center_floor_native': (initial_center + delta[:3] * [radius, radius, height]).tolist(),
                    'radius_native': float(radius * np.exp(delta[3])),
                    'height_native': float(height * np.exp(delta[4]))}
        trajectory = []
        def objective(delta):
            transform = pose(candidate(delta), rotation, origin)
            loss = float(np.mean([score_view(cylinder, transform, v)['loss'] for v in views]))
            trajectory.append({'delta': delta.tolist(), 'loss': loss})
            return loss
        initial_loss = objective(np.zeros(5))
        fit = minimize(objective, np.zeros(5), method='Nelder-Mead',
                       bounds=[(-1, 1), (-1, 1), (-.2, .2), (-np.log(2), np.log(2)), (-np.log(1.5), np.log(1.5))],
                       options={'maxiter': 100, 'xatol': .001, 'fatol': .0001,
                                'initial_simplex': np.vstack([np.zeros(5), np.diag([.06, .06, .015, .04, .025])])})
        fitted = candidate(fit.x) if fit.fun < initial_loss else initial
        transform = pose(fitted, rotation, origin)
        previous_path = root / f'result/{object_id}-comparison.json'
        previous = json.loads(previous_path.read_text())
        before = {v['frame_id']: v['generated_refined'] for v in previous['views']}
        per_view = [{'frame_id': v['frame_id'], 'before_generated_refined': before[v['frame_id']],
                     'after_cylinder': score_view(cylinder, transform, load_view(root, v['spec'], 288))} for v in views]
        comparisons.append({'object_id': object_id, 'baseline_file': str(previous_path.relative_to(root)),
                            'baseline_sha256': digest(previous_path), 'views': per_view})
        objects.append({'object_id': object_id, 'frame_ids': [v['frame_id'] for v in views],
                        'source_inventory_indices': source.get('source_inventory_indices', []),
                        'seed_parameters': initial, 'fitted_parameters': fitted,
                        'native_object_to_world': transform.tolist(), 'seed_evidence': seed_data,
                        'fitting': {'method': 'bounded Nelder-Mead; existing silhouette/depth loss',
                                    'iterations': int(fit.nit), 'evaluations': len(trajectory),
                                    'converged': bool(fit.success), 'termination': str(fit.message),
                                    'initial_loss': initial_loss, 'final_loss': float(min(fit.fun, initial_loss)),
                                    'fit_grid_height': 144, 'trajectory': trajectory},
                        'runtime_seconds': time.perf_counter() - obj_started})
        print(object_id, json.dumps({'parameters': fitted, 'mean_iou_before': np.mean([v['before_generated_refined']['visible_iou'] for v in per_view]),
                                    'mean_iou_after': np.mean([v['after_cylinder']['visible_iou'] for v in per_view])}), flush=True)
    write_json(output / 'parameters.json', {
        'run_id': 'lucida-replica-01', 'candidate': 'ground-axis-cylinder-64',
        'metric_scale_known': False, 'units': 'native Pi3X model-estimated units; not metres',
        'promoted_to_source_scene': False,
        'primitive': {'type': 'cylinder', 'radius': 1, 'height': 1, 'segments': 64, 'local_axis': '+Z', 'local_z_bounds': [-.5, .5]},
        'floor_normal_native': rotation[:, 2].tolist(), 'floor_frame_to_native_rotation': rotation.tolist(),
        'floor_frame_origin_native': origin.tolist(),
        'parameter_source': 'Frozen valid masked Pi3X points; robust body circle and z quantiles; numerical multiview refinement',
        'limitations': ['Circular cross-section and axis perpendicular to fitted floor are explicit shape priors.',
                       'No separate footplate, fasteners, dents or physical calibration.',
                       'All three views were used; metrics compare input consistency to estimated depth, not ground truth.',
                       'This candidate does not automatically replace the current generated scene.'],
        'source_files_sha256': {p: digest(root / p) for p in ['evidence/floor.json', 'evidence/objects.json', 'manifest.json']},
        'objects': objects, 'runtime_seconds': time.perf_counter() - started,
        'vlm_calls': 0, 'new_model_inference_calls': 0})
    decision_gate(comparisons)
    write_json(output / 'comparisons.json', {'run_id': 'lucida-replica-01', 'evaluation_grid_height': 288,
                                           'decision_gate': DECISION_GATE,
                                           'before': 'current generated asset with existing refined pose',
                                           'after': 'new fitted 64-segment cylinder', 'objects': comparisons})
    print(json.dumps(check(root)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--self-check', action='store_true')
    arguments = parser.parse_args()
    if arguments.self_check:
        print(json.dumps(check(arguments.root.resolve())))
    else:
        run(arguments.root.resolve())
