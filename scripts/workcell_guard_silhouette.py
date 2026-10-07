"""Left/right same-angle structural-prior model fitted to source silhouettes.

run(root, out, sources) requires exact A1 meshes in root/a1/guard-{left,right}.glb.
Generated surfaces initialize only. All objective residuals come from source SAM
masks projected through unchanged source cameras. This never measures truth.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from workcell_guard_joint import _inputs, _geometry
from workcell_photo_objects import _project
from ehs_spatial.platform.scene_measurements import fitted_bend
from ehs_spatial.platform.contracts import PlatformError

SIDES = ('left', 'right')


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _template(mesh):
    bend = fitted_bend(mesh.triangles)
    hinge = np.asarray(bend['hinge']); origin = hinge.mean(0)
    axis = hinge[1] - hinge[0]; axis /= np.linalg.norm(axis)
    rays = np.asarray([bend['rays'][0] - bend['rays'][1], bend['rays'][2] - bend['rays'][1]])
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    if np.cross(rays[0], rays[1]) @ axis < 0:
        axis = -axis
    rotation = np.column_stack([rays[0], np.cross(axis, rays[0]), axis])
    outlines = []
    for surface, ray in zip(bend['surfaces'], rays):
        points = surface['points'] - origin
        local = np.c_[np.maximum(0., points @ ray), points @ axis]
        local = np.vstack([local, [0, -np.linalg.norm(hinge[1] - hinge[0]) / 2], [0, np.linalg.norm(hinge[1] - hinge[0]) / 2]])
        hull = local[ConvexHull(local).vertices]
        polygon = cv2.approxPolyDP(hull.astype(np.float32), .008 * np.linalg.norm(np.ptp(hull, axis=0)), True).reshape(-1, 2)
        if len(polygon) < 3:
            raise ValueError('A1 surface outline is degenerate')
        outlines.append(polygon.astype(float))
    return {'origin': origin, 'rotation': rotation, 'outlines': outlines,
            'extent': float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))), 'initialAngleDeg': float(bend['value'])}


def _model(theta, parameters, template):
    origin = template['origin'] + parameters[:3] * template['extent']
    rotation = Rotation.from_rotvec(parameters[3:6]).as_matrix() @ template['rotation']
    pose = np.r_[origin, Rotation.from_matrix(rotation).as_rotvec(), theta]
    _, axis, directions, _ = _geometry(pose)
    scales = np.exp(parameters[6:9])
    polygons = [origin + outline[:, :1] * scales[index] * directions[index]
                + outline[:, 1:] * scales[2] * axis for index, outline in enumerate(template['outlines'])]
    return polygons, pose


def _render(polygons, frame, shape):
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        uv, depth = _project(polygon, frame)
        if np.any(depth <= 0) or not np.isfinite(uv).all():
            continue
        cv2.fillConvexPoly(mask, np.rint(uv * 256).astype(np.int32), 1, shift=8)
    return mask.astype(bool)


def _mesh_mask(mesh, frame, shape):
    uv, depth = _project(mesh.vertices, frame)
    mask = np.zeros(shape, np.uint8)
    xy = np.rint(uv * 256).astype(np.int32)
    for face in mesh.faces[np.all(depth[mesh.faces] > 0, axis=1)]:
        cv2.fillConvexPoly(mask, xy[face], 1, shift=8)
    return mask.astype(bool)


def _sample(points, maximum):
    return points[np.linspace(0, len(points) - 1, min(maximum, len(points))).astype(int)]


def _observations(boards, frames):
    observations = {}
    for side in SIDES:
        rows = []
        for photo, mask in sorted(boards[side]['masks'].items()):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            boundary = _sample(np.concatenate([c[:, 0] for c in contours]).astype(float), 120)
            inside = _sample(np.argwhere(mask)[:, ::-1].astype(float), 80)
            outer = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~mask
            outside = _sample(np.argwhere(outer)[:, ::-1].astype(float), 80)
            rows.append({'photo': photo, 'frame': frames[photo], 'mask': mask,
                         'boundary': boundary, 'inside': inside, 'outside': outside})
        observations[side] = rows
    return observations


def _signed_distance(points, polygon):
    """Continuous convex-polygon signed distance in source pixel coordinates."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edges = b - a; relative = points[:, None] - a
    t = np.clip(np.sum(relative * edges, axis=2) / np.maximum(np.sum(edges ** 2, axis=1), 1e-15), 0, 1)
    distance = np.sqrt(np.min(np.sum((relative - t[..., None] * edges) ** 2, axis=2), axis=1))
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    inside = np.all(cross >= -1e-9, axis=1) | np.all(cross <= 1e-9, axis=1)
    return np.where(inside, -distance, distance)


def _residual(values, templates, observations, deadline=np.inf):
    if time.monotonic() > deadline:
        raise TimeoutError('Bounded silhouette optimization time exhausted')
    residual = []
    for index, side in enumerate(SIDES):
        polygons, _ = _model(values[0], values[1 + 9 * index:10 + 9 * index], templates[side])
        for row in observations[side]:
            projected = [_project(polygon, row['frame']) for polygon in polygons]
            n = sum(len(row[key]) for key in ('boundary', 'inside', 'outside'))
            if any(np.any(depth <= 0) or not np.isfinite(uv).all() for uv, depth in projected):
                residual.append(np.full(n, 1000.)); continue
            for kind in ('boundary', 'inside', 'outside'):
                distance = np.minimum(*[_signed_distance(row[kind], uv) for uv, _ in projected])
                if kind == 'inside':
                    distance = 2 * np.maximum(distance, 0)
                elif kind == 'outside':
                    distance = 2 * np.minimum(distance, 0)
                residual.append(distance)
    return np.concatenate(residual)


def _metrics(values, templates, observations, meshes=None):
    result = {}
    for index, side in enumerate(SIDES):
        polygons, _ = _model(values[0], values[1 + 9 * index:10 + 9 * index], templates[side])
        rows = []
        for row in observations[side]:
            predicted = (_mesh_mask(meshes[side], row['frame'], row['mask'].shape) if meshes is not None
                         else _render(polygons, row['frame'], row['mask'].shape))
            iou = float((predicted & row['mask']).sum() / max(1, (predicted | row['mask']).sum()))
            boundary = cv2.morphologyEx(predicted.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            distance = cv2.distanceTransform((1 - boundary).astype(np.uint8), cv2.DIST_L2, 5)
            xy = np.rint(row['boundary']).astype(int)
            rows.append({'photo': row['photo'], 'iou': iou,
                         'sourceBoundaryMedianPx': float(np.median(distance[xy[:, 1], xy[:, 0]]))})
        result[side] = {'views': rows, 'meanIoU': float(np.mean([r['iou'] for r in rows])),
                        'medianIoU': float(np.median([r['iou'] for r in rows]))}
    return result


def _gate(candidate, reference):
    reasons = []
    for side in SIDES:
        if candidate[side]['meanIoU'] < reference[side]['meanIoU'] - .02:
            reasons.append(f'{side}: mean IoU worsens by more than 0.02')
        if candidate[side]['medianIoU'] < reference[side]['medianIoU'] - .02:
            reasons.append(f'{side}: median IoU worsens by more than 0.02')
        old = {r['photo']: r['iou'] for r in reference[side]['views']}
        for row in candidate[side]['views']:
            if row['iou'] < old[row['photo']] - .08:
                reasons.append(f'{side} photo {row["photo"]}: IoU worsens by more than 0.08')
    return {'passed': not reasons, 'reasons': reasons}


def _solve(initial, templates, observations, deadline, fixed_theta=None, max_nfev=65):
    board_lower = [-.6] * 3 + [-.65] * 3 + list(np.log([.5, .5, .65]))
    board_upper = [.6] * 3 + [.65] * 3 + list(np.log([1.8, 1.8, 1.45]))
    lower = np.r_[np.deg2rad(20), board_lower, board_lower]
    upper = np.r_[np.deg2rad(175), board_upper, board_upper]
    expand = (lambda p: np.r_[fixed_theta, p]) if fixed_theta is not None else (lambda p: p)
    x0 = initial[1:] if fixed_theta is not None else initial
    fit = least_squares(lambda p: _residual(expand(p), templates, observations, deadline), x0,
                        bounds=(lower[1:], upper[1:]) if fixed_theta is not None else (lower, upper),
                        loss='soft_l1', f_scale=1.5, x_scale='jac', max_nfev=max_nfev,
                        ftol=2e-5, xtol=2e-5, gtol=2e-5)
    values = expand(fit.x)
    return values, {'converged': bool(fit.success), 'evaluations': fit.nfev,
                    'residualRmsPx': float(np.sqrt(np.mean(_residual(values, templates, observations) ** 2))),
                    'cost': float(fit.cost), 'commonAngleDeg': float(np.rad2deg(values[0]))}


def _export(polygons, pose, board, frames, path):
    scene = trimesh.Scene(); color_fractions = []
    for index, polygon in enumerate(polygons):
        mesh = trimesh.Trimesh(vertices=polygon, faces=[[0, j, j + 1] for j in range(1, len(polygon) - 1)], process=False)
        for _ in range(4):
            mesh = mesh.subdivide()
        colors = np.full((len(mesh.vertices), 4), [115, 115, 115, 255], np.uint8)
        colored = np.zeros(len(mesh.vertices), bool)
        normal = _geometry(pose)[3][index]
        photos = sorted(board['masks'], key=lambda p: -abs(normal @ (frames[p]['pose'][:3, 3] - polygon.mean(0))) /
                        np.linalg.norm(frames[p]['pose'][:3, 3] - polygon.mean(0)))
        for photo in photos:
            frame, mask = frames[photo], board['masks'][photo]
            uv, depth = _project(mesh.vertices, frame); xy = np.rint(uv).astype(int)
            valid = (~colored) & (depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
            ids = np.flatnonzero(valid); ids = ids[mask[xy[ids, 1], xy[ids, 0]]]
            pixel = np.rint((np.c_[uv[ids], np.ones(len(ids))] @ frame['C'].T)[:, :2]).astype(int)
            image = frame['analysisRgb']; pixel[:, 0] = pixel[:, 0].clip(0, image.shape[1] - 1); pixel[:, 1] = pixel[:, 1].clip(0, image.shape[0] - 1)
            colors[ids, :3] = image[pixel[:, 1], pixel[:, 0]]; colored[ids] = True
        mesh.visual.vertex_colors = colors
        mesh.visual.material = trimesh.visual.material.PBRMaterial(doubleSided=True, baseColorFactor=[255, 255, 255, 255],
                                                                  metallicFactor=0., roughnessFactor=.8)
        scene.add_geometry(mesh, node_name=f'v-guard-{board["side"]}-panel-{index + 1}', geom_name=f'panel-{index + 1}')
        color_fractions.append(float(colored.mean()))
    scene.export(path)
    loaded = trimesh.load(path, force='scene')
    world_meshes = []
    for node in loaded.graph.nodes_geometry:
        transform, geometry = loaded.graph[node]
        mesh = loaded.geometry[geometry].copy(); mesh.apply_transform(transform); world_meshes.append(mesh)
    if len(world_meshes) != 2:
        raise ValueError('Export must contain exactly two analytic sheets')
    centers = np.array([mesh.vertices.mean(0) for mesh in world_meshes])
    normals = np.array([np.linalg.svd(mesh.vertices - center, full_matrices=False)[2][-1] for mesh, center in zip(world_meshes, centers)])
    axis = np.cross(*normals); axis /= np.linalg.norm(axis)
    origin = np.linalg.solve(np.vstack([normals, axis]), [normals[0] @ centers[0], normals[1] @ centers[1], axis @ centers.mean(0)])
    rays = centers - origin; rays -= np.outer(rays @ axis, axis); rays /= np.linalg.norm(rays, axis=1)[:, None]
    angle = float(np.rad2deg(np.arccos(np.clip(rays[0] @ rays[1], -1, 1))))
    if abs(angle - np.rad2deg(pose[6])) > .002:
        raise ValueError('Exported mesh and shared parameter disagree')
    try:
        annotation = {'supported': True, 'angleDeg': float(fitted_bend(np.concatenate([m.triangles for m in world_meshes]))['value'])}
    except PlatformError as error:
        annotation = {'supported': False, 'reason': str(error)}
    return {'meshAngleReadbackDeg': angle, 'photoColorFractionPerPanel': color_fractions,
            'nodes': list(loaded.graph.nodes_geometry), 'fittedBendAnnotation': annotation, 'doubleSided': True}


def run(root, out, sources):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    if len(sources) < 2:
        raise ValueError('Every raw source photo of the scene (at least two) is required')
    started = time.monotonic()
    meshes = {side: trimesh.load(root / 'a1' / f'guard-{side}.glb', force='mesh') for side in SIDES}
    templates, unsupported = {}, []
    for side, mesh in meshes.items():
        try:
            templates[side] = _template(mesh)
        except PlatformError as error:
            unsupported.append(f'{side}: {error.code}')
    if unsupported:
        # Generated A1 boards without a stable two-face bend cannot initialize the shared-angle model:
        # A1 stays in the scene and this records why (never a failed run, never a fabricated angle).
        result = {'schemaVersion': 1, 'route': 'shared-angle-silhouette-prior', 'objects': [], 'overlays': [],
                  'promotionAllowed': False, 'selected': False, 'status': 'unsupported-initializer',
                  'fitGate': {'passed': False, 'reasons': ['A1 board has no stable two-face bend to initialize the shared-angle model: ' + '; '.join(unsupported)]},
                  'centerPolicy': 'unchanged A1', 'prior': 'User supplied same-angle specification for left and right; not applied.',
                  'wallSeconds': time.monotonic() - started}
        _write(out / 'results.json', result)
        return result
    frames, boards, association = _inputs(root, sources=sources)
    observations = _observations(boards, frames)
    initial = np.r_[np.deg2rad(templates['left']['initialAngleDeg']), np.zeros(18)]
    baseline = _metrics(initial, templates, observations, meshes)
    analytic_initial = _metrics(initial, templates, observations)
    deadline = time.monotonic() + 180.
    starts = [initial[0], *np.deg2rad([50, 90, 130, 165])]
    solutions, attempts = [], []
    def solve_start(theta):
        tick = time.monotonic(); x = initial.copy(); x[0] = theta
        try:
            values, diagnostic = _solve(x, templates, observations, deadline)
            metrics = _metrics(values, templates, observations)
            diagnostic.update(startAngleDeg=float(np.rad2deg(theta)), seconds=time.monotonic() - tick,
                              sourceFit=metrics, fitGate=_gate(metrics, baseline))
            return values, diagnostic
        except TimeoutError as error:
            return None, {'startAngleDeg': float(np.rad2deg(theta)), 'error': str(error)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for values, diagnostic in pool.map(solve_start, starts):
            attempts.append(diagnostic)
            if values is not None:
                solutions.append((values, diagnostic))
    if not solutions:
        raise RuntimeError('No bounded source-silhouette optimization completed')
    eligible = [s for s in solutions if s[1]['fitGate']['passed'] and s[1]['converged']]
    values, best = min(eligible or solutions, key=lambda s: s[1]['cost'])
    final_metrics = best['sourceFit']; gate = _gate(final_metrics, baseline)
    if not best['converged']:
        gate['passed'] = False; gate['reasons'].append('selected optimizer did not converge')
    profile = []
    for angle in sorted(set([30., 60., 90., 120., 150., 174., float(np.rad2deg(values[0]))])):
        if time.monotonic() > deadline:
            profile.append({'angleDeg': angle, 'status': 'not-run-time-budget'}); continue
        try:
            profiled, diagnostic = _solve(values, templates, observations, deadline, fixed_theta=np.deg2rad(angle), max_nfev=40)
            metrics = _metrics(profiled, templates, observations)
            compatible = diagnostic['converged'] and _gate(metrics, baseline)['passed'] and _gate(metrics, final_metrics)['passed'] and diagnostic['residualRmsPx'] <= best['residualRmsPx'] + .25
            profile.append({'angleDeg': angle, 'status': 'fit' if diagnostic['converged'] else 'unconverged', 'imageCompatible': compatible,
                            'optimizer': diagnostic, 'sourceFit': metrics})
        except TimeoutError:
            profile.append({'angleDeg': angle, 'status': 'not-run-time-budget'})
    compatible = [r['angleDeg'] for r in profile if r.get('imageCompatible')]
    ambiguity = {'kind': 'conditional profile sensitivity; not a statistical confidence interval',
                 'profileComplete': all(r['status'] == 'fit' for r in profile),
                 'imageCompatibleProfileAnglesDeg': compatible,
                 'ambiguous': bool(len(compatible) < 2 or np.ptp(compatible) >= 10. or any(r['status'] != 'fit' for r in profile)),
                 'scope': 'Finite masks/cameras and bounded template poses/dimensions can admit additional untested shapes.'}
    objects = []
    for index, side in enumerate(SIDES):
        polygons, pose = _model(values[0], values[1 + 9 * index:10 + 9 * index], templates[side])
        file = f'guard-{side}.glb'
        exported = _export(polygons, pose, boards[side], frames, out / file)
        if not exported['fittedBendAnnotation']['supported']:
            gate['passed'] = False; gate['reasons'].append(f'{side}: existing viewer bend annotation cannot support exported sheets')
        objects.append({'id': f'v-guard-{side}', 'mesh': file,
                        'status': 'prior-constrained-model' if gate['passed'] else 'rejected-source-fit',
                        'measurementAngleDeg': None, 'modelAngleDeg': float(np.rad2deg(values[0])), **exported,
                        'sourceFit': final_metrics[side], 'a1SourceFit': baseline[side], 'analyticInitialSourceFit': analytic_initial[side],
                        'candidateGeometry': {'originNative': pose[:3].tolist(), 'rotationVector': pose[3:6].tolist(),
                                              'panelPolygonsNative': [p.tolist() for p in polygons],
                                              'widthHeightScales': np.exp(values[1 + 9 * index:10 + 9 * index][6:9]).tolist()}})
    for row in objects:
        row['status'] = 'prior-constrained-model' if gate['passed'] else 'rejected-source-fit'
        row['meshRole'] = 'prior-constrained-model' if gate['passed'] else 'diagnostic-candidate'
    for photo in frames:
        image = frames[photo]['rgb'].copy()
        for index, side in enumerate(SIDES):
            if photo not in boards[side]['masks']:
                continue
            target = boards[side]['masks'][photo]
            polygons, _ = _model(values[0], values[1 + 9 * index:10 + 9 * index], templates[side])
            for mask, color in ((target, (30, 230, 60)), (_mesh_mask(meshes[side], frames[photo], target.shape), (230, 40, 180)),
                                (_render(polygons, frames[photo], target.shape), (30, 210, 240))):
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 1)
        cv2.imwrite(str(out / f'overlay-{photo}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    result = {'schemaVersion': 1, 'route': 'shared-angle-silhouette-prior', 'objects': objects,
              'promotionAllowed': gate['passed'], 'selected': gate['passed'],
              'status': 'prior-constrained-model' if gate['passed'] else 'rejected-source-fit',
              'fitGate': gate, 'sharedAngleDeg': float(np.rad2deg(values[0])),
              'fitGateCriteria': {'maximumMeanIoUDrop': .02, 'maximumMedianIoUDrop': .02, 'maximumPerViewIoUDrop': .08,
                                  'reference': 'exact A1 source silhouette fit, per board; convergence and viewer annotation support also required'},
              'prior': 'User supplied same-angle specification for left and right; exact equality is imposed, never accuracy evidence.',
              'centerPolicy': 'unchanged A1; excluded from this objective and all shared parameters',
              'coordinateSystem': 'unchanged source MapAnything world and cameras', 'association': association,
              'initialization': {'source': 'exact root/a1 meshes; model surfaces and convex outlines only',
                                 'commonInitialAngleSource': 'left A1 model angle; not an average or measured prior',
                                 'a1ModelAnglesDeg': {s: templates[s]['initialAngleDeg'] for s in SIDES}},
              'optimizedVariables': 'one common theta plus independent rigid pose, two panel widths and hinge-height scale per board',
              'objective': 'source mask contour signed distance plus foreground/background silhouette coverage; no generated-mesh residual',
              'a1SourceFit': baseline, 'analyticInitialSourceFit': analytic_initial, 'finalSourceFit': final_metrics,
              'multistarts': attempts, 'profile': profile, 'ambiguity': ambiguity,
              'overlays': [f'overlay-{p}.jpg' for p in frames], 'overlayLegend': 'green: source mask; magenta: A1; cyan: prior-constrained candidate',
              'limitations': ['Model angle is constrained/inferred, not measured physical truth.',
                              'Silhouettes do not uniquely identify fold angle, hidden shape, thickness or camera error.',
                              'Convex template outlines originate from generated A1 geometry; source masks determine pose/size/theta optimization.',
                              'Masks describe visible segmentation and may contain occlusion or association error; no stripe is interpreted as a crease.',
                              'Zero-thickness planar sheets; source-colored visible regions and gray unseen regions; no absolute accuracy claim.'],
              'wallSeconds': time.monotonic() - started}
    _write(out / 'results.json', result)
    return result


def _self_check():
    import struct
    templates = {side: {'origin': np.array([x, 0., 0.]), 'rotation': np.eye(3), 'extent': 1.,
                        'outlines': [np.array([[0, -.5], [.7, -.5], [.7, .5], [0, .5]])] * 2}
                 for side, x in zip(SIDES, (-1., 1.))}
    frames = {}
    for photo, center in enumerate(([0, 5, 2], [-3, 4, 2], [3, 4, 1]), 1):
        z = -np.asarray(center, float); z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x)
        pose = np.eye(4); pose[:3, :3] = np.column_stack([x, np.cross(z, x), z]); pose[:3, 3] = center
        frames[photo] = {'K': np.array([[220., 0, 160], [0, 220., 120], [0, 0, 1.]]), 'pose': pose,
                         'rgb': np.full((240, 320, 3), 180, np.uint8), 'analysisRgb': np.full((240, 320, 3), 180, np.uint8), 'C': np.eye(3)}
    recovered = []
    for truth in (110., 125.):
        target = np.r_[np.deg2rad(truth), np.zeros(18)]
        boards = {side: {'side': side, 'masks': {p: _render(_model(target[0], np.zeros(9), templates[side])[0], f, (240, 320)) for p, f in frames.items()}} for side in SIDES}
        observations = _observations(boards, frames)
        fit = least_squares(lambda theta: _residual(np.r_[theta, np.zeros(18)], templates, observations), [np.deg2rad(150)],
                            bounds=(np.deg2rad([20]), np.deg2rad([175])), max_nfev=35)
        angle = float(np.rad2deg(fit.x[0])); recovered.append(angle)
        assert abs(angle - truth) < 3., (angle, truth)
        with tempfile.TemporaryDirectory() as directory:
            angles = []
            for side in SIDES:
                parameters = np.r_[.1, -.05, .07, .12, -.08, .05, np.log([1.1, .9, 1.05])]
                polygons, pose = _model(fit.x[0], parameters, templates[side])
                exported = _export(polygons, pose, boards[side], frames, Path(directory) / f'{side}.glb')
                angles.append(exported['meshAngleReadbackDeg'])
                blob = (Path(directory) / f'{side}.glb').read_bytes()
                tree = json.loads(blob[20:20 + struct.unpack('<I', blob[12:16])[0]])
                assert all(material['doubleSided'] for material in tree['materials'])
                assert all('COLOR_0' in primitive['attributes'] for mesh in tree['meshes'] for primitive in mesh['primitives'])
                origin, axis, directions, _ = _geometry(pose)
                for index, polygon in enumerate(polygons):
                    expected = origin + templates[side]['outlines'][index][:, :1] * np.exp(parameters[6 + index]) * directions[index] + templates[side]['outlines'][index][:, 1:] * np.exp(parameters[8]) * axis
                    assert np.allclose(polygon, expected)
            assert np.ptp(angles) < .002
            for angle in (25., 175.):
                polygons, pose = _model(np.deg2rad(angle), np.zeros(9), templates['left'])
                exported = _export(polygons, pose, boards['left'], frames, Path(directory) / f'angle-{angle}.glb')
                assert abs(exported['meshAngleReadbackDeg'] - angle) < .002
                if angle == 175.:
                    assert not exported['fittedBendAnnotation']['supported']
    assert recovered[1] - recovered[0] > 10., 'Changing source silhouettes must change the fit despite the same 150-degree initializer'
    print('PASS: source silhouette response, shared/acute/near-flat actual GLB angles, rigid/dimension transforms, double-sided vertex colors and separate viewer-annotation support')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--images', type=Path, nargs='+', help='Every photo of the scene, in photo order')
    args = parser.parse_args()
    if args.self_check:
        _self_check()
    elif args.root is None or args.out is None or args.images is None:
        parser.error('--root, --out and every scene photo as --images are required')
    else:
        result = run(args.root, args.out, args.images)
        print(json.dumps({k: result[k] for k in ('sharedAngleDeg', 'promotionAllowed', 'fitGate', 'ambiguity', 'wallSeconds')}, indent=2))
