"""Place generated object assets in the observed scene and evaluate real views.

All available object views constrain pose refinement, with each view reported
separately. Native generated assets and their initial poses are kept.
"""

import hashlib
import html
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import trimesh


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(parts):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', parts['rotation_deg'], degrees=True).as_matrix() @ np.diag(parts['scale'])
    result[:3, 3] = parts['position']
    return result


def decompose(transform):
    transform = np.asarray(transform, float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError('Expected a finite object-to-world affine matrix')
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    if np.any(scale <= 0):
        raise ValueError('Object scale must be positive')
    rotation = transform[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < .999:
        raise ValueError('Object transform has shear or reflection; do not silently discard it')
    value = {'position': transform[:3, 3].tolist(), 'rotation_deg': Rotation.from_matrix(rotation).as_euler('xyz', degrees=True).tolist(), 'scale': scale.tolist()}
    if not np.allclose(matrix(value), transform, atol=1e-4):
        raise ValueError('Transform decomposition changed the asset pose')
    return value


def transformed(vertices, transform):
    return np.asarray(vertices) @ transform[:3, :3].T + transform[:3, 3]


def native_view(root, spec):
    image = np.asarray(Image.open(root / spec['canonical_rgb_path']).convert('RGB'))
    points = np.load(root / spec['pointmap_path'], allow_pickle=False)
    K = np.load(root / spec['K_path'], allow_pickle=False)
    c2w = np.load(root / spec['c2w_path'], allow_pickle=False)
    if points.shape != (*image.shape[:2], 3):
        raise ValueError('Image and native point map must share the exact grid')
    if (K.shape != (3,3) or c2w.shape != (4,4) or not np.isfinite(K).all()
            or not np.isfinite(c2w).all() or not np.allclose(c2w[3],[0,0,0,1])
            or not np.allclose(c2w[:3,:3].T@c2w[:3,:3],np.eye(3),atol=1e-4)):
        raise ValueError('Invalid native camera matrices')
    local = transformed(points, np.linalg.inv(c2w))
    valid = np.isfinite(local).all(-1) & (local[..., 2] > 0)
    if spec.get('valid_path'):
        mask=np.load(root / spec['valid_path'], allow_pickle=False)
        if mask.shape!=image.shape[:2]:raise ValueError('Validity and image grids differ')
        valid &= mask.astype(bool)
    if spec.get('content_valid_path'):
        mask=np.load(root / spec['content_valid_path'], allow_pickle=False)
        if mask.shape!=image.shape[:2]:raise ValueError('Content and image grids differ')
        valid &= mask.astype(bool)
    if 'content_rect_xyxy' in spec:
        rect=spec['content_rect_xyxy'];h,w=image.shape[:2]
        if (not isinstance(rect,list) or len(rect)!=4 or any(type(v) is not int for v in rect)
                or not 0<=rect[0]<rect[2]<=w or not 0<=rect[1]<rect[3]<=h):
            raise ValueError('Recorded canonical content rectangle is missing or invalid')
        content=np.zeros((h,w),bool);content[rect[1]:rect[3],rect[0]:rect[2]]=True
        valid &= content
    if spec.get('conf_path'):
        confidence=np.load(root / spec['conf_path'], allow_pickle=False)
        if confidence.shape!=image.shape[:2]:raise ValueError('Confidence and image grids differ')
        valid &= np.isfinite(confidence) & (confidence >= .1)
    return dict(spec=spec, frame_id=spec['frame_id'], rgb=image, K=K, c2w=c2w,
                points=points, valid=valid, native_depth=local[..., 2])


def load_view(root, spec, size=144):
    view = native_view(root, spec)
    image, K, c2w, valid = (view[k] for k in ['rgb','K','c2w','valid'])
    mask = np.load(root / spec['canonical_mask_path'], allow_pickle=False).astype(bool)
    if mask.shape != image.shape[:2]:
        raise ValueError('Image, mask and native point map must share the exact grid')
    height, width = mask.shape
    small_width = max(16, round(width * size / height))
    scale = np.array([small_width / width, size / height])
    small_K = K.copy()
    small_K[0] *= scale[0]
    small_K[1] *= scale[1]
    small_K[:2, 2] += (scale - 1) / 2
    depth = np.where(valid, view['native_depth'], 0).astype(np.float32)
    small_depth = cv2.resize(depth, (small_width, size), interpolation=cv2.INTER_NEAREST_EXACT)
    small_mask = cv2.resize(mask.astype(np.uint8), (small_width, size), interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
    yy, xx = np.indices(small_mask.shape)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(small_K).T @ c2w[:3, :3].T
    rays = np.concatenate([np.broadcast_to(c2w[:3, 3], rays.shape), rays], axis=-1).astype(np.float32)
    if (small_mask & (small_depth > 0)).sum() < 8:
        raise ValueError('Too little visible object geometry for placement scoring')
    return dict(view, mask=mask, depth=small_depth, target=small_mask, rays=rays)


def scoring_height(root, spec, requested):
    """Keep eight real depth samples; a small object needs a denser scoring grid."""
    mask = np.load(root / spec['canonical_mask_path'], allow_pickle=False).astype(bool)
    valid = np.load(root / (spec.get('content_valid_path') or spec['valid_path']), allow_pickle=False).astype(bool)
    points = np.load(root / spec['pointmap_path'], allow_pickle=False)
    c2w = np.load(root / spec['c2w_path'], allow_pickle=False)

    _, valid = camera_depth(points, c2w, valid)
    conf = np.load(root / spec['conf_path'], allow_pickle=False) if 'conf_path' in spec else np.ones(mask.shape)
    supported = mask & valid & np.isfinite(conf) & (conf >= .1)
    height, width = mask.shape
    size = min(requested, height)
    while True:
        sampled = cv2.resize(supported.astype(np.uint8), (max(16, round(width*size/height)), size), interpolation=cv2.INTER_NEAREST_EXACT)
        if sampled.sum() >= 8:
            return size
        if size == height:
            raise ValueError('Fewer than eight native depth samples; placement cannot be scored')
        size = min(height, size * 2)


def cast_depth(mesh, transform, rays):
    # Mesh vertices stay immutable throughout this experiment. Transform rays
    # into object coordinates so every pose can reuse the same spatial index.
    if not hasattr(mesh, '_research_ray_scene'):
        mesh._research_ray_scene = o3d.t.geometry.RaycastingScene()
        mesh._research_ray_scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                                               o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    inverse = np.linalg.inv(transform)
    local_rays = np.concatenate([transformed(rays[..., :3], inverse),
                                 rays[..., 3:] @ inverse[:3, :3].T], axis=-1).astype(np.float32)
    # Do not normalise directions: preserving their parameter keeps t_hit in
    # the source camera's z units even under nonuniform object scale.
    return mesh._research_ray_scene.cast_rays(o3d.core.Tensor(local_rays))['t_hit'].numpy()


def score_view(mesh, transform, view):
    predicted = cast_depth(mesh, transform, view['rays'])
    target, depth = view['target'], view['depth']
    finite = np.isfinite(predicted) & (predicted > 0)
    # Observed foreground may hide a complete object's inferred surfaces. Never
    # hide a misaligned prediction inside the target mask to improve its score.
    occluded = (~target) & (depth > 0) & (predicted > depth * 1.04)
    visible = finite & ~occluded
    overlap = visible & target
    union = visible | target
    iou = float(overlap.sum() / max(1, union.sum()))
    supported = overlap & (depth > 0)
    residual = np.abs(predicted[supported] - depth[supported]) / depth[supported]
    p50, p95 = (np.percentile(residual, [50, 95]).tolist() if len(residual) else [None, None])
    def edge(mask):
        return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    a, b = edge(visible), edge(target)
    if a.any() and b.any():
        da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
        db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
        boundary = float((np.mean(da[b]) + np.mean(db[a])) / (2 * len(target)))
    else:
        boundary = 1.0
    return {'visible_iou': iou, 'boundary_error_image_height': boundary,
            'relative_depth_p50': p50, 'relative_depth_p95': p95,
            'target_pixels': int(target.sum()), 'visible_predicted_pixels': int(visible.sum()),
            'depth_comparison_pixels': int(supported.sum()),
            'loss': float(1 - iou + 2 * boundary + min(p50 if p50 is not None else 1, 1))}


FLOOR_CONTACT_WEIGHT = 2.0


def floor_penalty(vertices, plane, weight=FLOOR_CONTACT_WEIGHT):
    """One floor for every object, no per-object rules: a hinge on the lowest point (0.5th-percentile height, the report's
    'lowest point') below the floor plane, normalised by the object's own height so it is unit-free. Nothing pulls an object
    down: mounted objects (robot, guard, light curtains) may float; nothing may sink. plane = [nx, ny, nz, d], height = X.n + d."""
    h = vertices @ plane[:3] + plane[3]
    lowest = float(np.percentile(h, 0.5))
    return weight * max(0.0, -lowest) / max(float(np.ptp(h)), 1e-6), lowest


def refine(mesh, initial, views, max_iterations=100, floor=None):
    extent = np.ptp(transformed(mesh.vertices, initial), axis=0)
    radius = max(float(np.linalg.norm(extent)), 1e-4)
    floor = None if floor is None else np.asarray(floor, float)
    initial_parts = decompose(initial)
    initial_rotation = Rotation.from_euler('xyz', initial_parts['rotation_deg'], degrees=True).as_matrix()
    initial_scale = np.array(initial_parts['scale'])
    def candidate(x):
        result = initial.copy()
        result[:3, :3] = initial_rotation @ Rotation.from_rotvec(x[3:6]).as_matrix() @ np.diag(initial_scale * np.exp(x[6:9]))
        result[:3, 3] += x[:3] * radius
        return result
    def score(transform):
        per_view = {view['frame_id']: score_view(mesh, transform, view) for view in views}
        penalty, lowest = floor_penalty(transformed(mesh.vertices, transform), floor) if floor is not None else (0.0, None)
        return {'loss': float(np.mean([value['loss'] for value in per_view.values()]) + penalty), 'views': per_view,
                'floor': {'penalty': penalty, 'lowest_native': lowest}}
    initial_score = score(initial)
    records = []
    def objective(x):
        # ponytail: bounded local refinement starts from the learned pose. This
        # does not solve arbitrary axis permutations or replace GizmoAct.
        if np.max(np.abs(x[:3])) > .3 or np.linalg.norm(x[3:6]) > .65 or np.max(np.abs(x[6:])) > .4:
            return 100.0 + float(np.dot(x, x))
        result = score(candidate(x))
        records.append({'delta': x.tolist(), **result})
        return result['loss']
    simplex = np.zeros((10, 9))
    simplex[1:] = np.diag([.015] * 3 + [.04] * 3 + [.04] * 3)
    start = time.perf_counter()
    fit = minimize(objective, np.zeros(9), method='Nelder-Mead', options={
        'initial_simplex': simplex, 'maxiter': max_iterations, 'xatol': .002, 'fatol': .001})
    final = candidate(fit.x) if fit.fun < initial_score['loss'] else initial.copy()
    # No transform with shear may enter the editable scene contract.
    decompose(final)
    return final, {'method': 'bounded multi-view CPU render-and-compare; not GizmoAct', 'training_frames': [v['frame_id'] for v in views],
                   'floor_contact': None if floor is None else {'weight': FLOOR_CONTACT_WEIGHT, 'plane_native': floor.tolist(),
                                                                 'rule': 'hinge on the 0.5th-percentile height below the floor, / object height; sinking only'},
                   'seconds': time.perf_counter() - start, 'evaluations': len(records),
                   'initial': initial_score, 'final': score(final), 'trajectory': records}


def observed_mesh(view, *, return_pixel_faces=False):
    """Triangulate only neighbouring observed pixels, retaining real holes."""
    points = view['points']
    height, width = points.shape[:2]
    grid = np.arange(height * width).reshape(height, width)
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                            np.stack([b, d, c], -1).reshape(-1, 3)])
    valid = (view['mask'] & view['valid']).ravel()
    faces = faces[valid[faces].all(axis=1)]
    vertices = points.reshape(-1, 3)
    # ponytail: a local depth-relative edge bound avoids bridging discontinuities;
    # this is an observed surface, without unobserved back-side completion.
    z = transformed(vertices, np.linalg.inv(view['c2w']))[:, 2]
    lengths = np.linalg.norm(vertices[faces] - vertices[faces[:, [1, 2, 0]]], axis=2)
    faces = faces[(lengths.max(axis=1) <= .04 * np.median(z[faces], axis=1))]
    if not len(faces):
        raise ValueError('No supported triangles in observed surface')
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                           vertex_colors=view['rgb'].reshape(-1, 3), process=False)
    mesh.remove_unreferenced_vertices()
    return (mesh, faces) if return_pixel_faces else mesh


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_metrics(output, run_id, comparisons, unavailable):
    sections = []
    for comparison in comparisons:
        rows = []
        for view in comparison['views']:
            before, after = view['generated_initial'], view['generated_refined']
            cells = [html.escape(view['frame_id']),
                     'anchor view / used for refinement' if view['role'] == 'anchor; used for pose refinement' else 'other view / used for refinement']
            for metric in ['visible_iou', 'boundary_error_image_height', 'relative_depth_p50']:
                cells.extend('no overlap' if value is None else f'{value * 100:.2f}%' for value in [before[metric], after[metric]])
            cells.append(f"{view['observed_surface']['visible_iou'] * 100:.2f}%")
            rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
        sections.append(f'<section><h2>{html.escape(comparison["label"])}</h2>'
            f'<p>{comparison["faces"]:,} triangles · watertight mesh：{"yes" if comparison["watertight"] else "no"} · '
            f'<a href="{html.escape(comparison["object_id"])}-comparison.json">complete record</a></p>'
            '<div class="scroll"><table><thead><tr><th>input photo</th><th>role</th>'
            '<th>silhouette IoU before ↑</th><th>after ↑</th><th>boundary distance before ↓</th><th>after ↓</th>'
            '<th>relative depth error before ↓</th><th>after ↓</th><th>observed-surface IoU</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
            f'<img loading="lazy" style="width:100%;height:auto;margin-top:18px" '
            f'src="{html.escape(comparison["object_id"])}-alignment.png" '
            f'alt="{html.escape(comparison["label"])}: input segmentation and before/after projections"></section>')
    missing = (f'<p>The remaining {len(unavailable)} candidates are unavailable. Per-object states are in the <a href="comparisons.json">complete record</a>。</p>') if unavailable else ''
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>Workcell reconstruction · same-view comparison</title><style>body{font:15px/1.6 system-ui;margin:0;background:#f3f4ef;color:#253b38}main{max-width:1280px;margin:40px auto;padding:0 24px}h1{font-size:30px}h2{font-size:20px}p{max-width:960px}section{background:white;border:1px solid #d3dcd4;border-radius:12px;padding:20px;margin:22px 0}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;text-align:right;border-bottom:1px solid #e0e6e1}td:first-child,th:first-child{text-align:left}a{color:#176c62}</style><main>'
    document += f'<h1>The same photos, objects and metrics</h1><p>Run: {html.escape(run_id)}</p>'
    document += '<p>Before is the generated mesh in its predicted pose; after is the same mesh with local placement refined against all supplied object views. Objects and photos are unchanged. Boundary distance is normalized by image height. Median relative depth error compares against estimated source depth and does not establish physical measurement accuracy.</p>'
    document += '<p>Observed surfaces contain only photographed triangles. Each object view is listed separately. These scores measure input consistency. Watertightness describes mesh topology and does not verify hidden shapes.</p>'
    document += '<p>Assets are static editable meshes. Robot joints and physical collisions have not been validated.</p>' + missing + ''.join(sections)
    document += '<p><a href="comparisons.json">Download all metrics</a> · <a href="scene.glb">Download GLB</a></p></main></html>'
    (output / 'metrics.html').write_text(document)


def mesh_bytes(mesh):
    visual = mesh.visual.to_color() if mesh.visual.kind == 'texture' else mesh.visual
    colors = visual.vertex_colors[:, :3].astype(np.float32) / 255
    indices = np.asarray(mesh.faces, dtype='<u4').ravel()
    vertices = np.column_stack([mesh.vertices, mesh.vertex_normals, colors]).astype('<f4')
    if not np.isfinite(vertices).all():
        raise ValueError('Nonfinite viewer geometry')
    return vertices.tobytes()+indices.tobytes(), {'byte_offset':0,'vertex_count':len(vertices),'stride':9,
        'index_byte_offset':vertices.nbytes,'index_count':len(indices),'index_type':'uint32'}


def assemble(root, iterations=100, eval_size=288):
    root = root.resolve()
    output = root / 'result'
    output.mkdir(exist_ok=True)
    source = json.loads((root / 'manifest.json').read_text())
    evidence = json.loads((root / 'evidence/objects.json').read_text())
    floor = json.loads((root / 'evidence/floor.json').read_text())
    entries, assets, comparisons, unavailable = [], [], [], []
    for obj in evidence['objects']:
        object_id = obj['object_id']
        record_path = root / 'generation' / object_id / 'output.json'
        if not record_path.exists():
            unavailable.append({'id': object_id, 'reason': 'No native model output'})
            continue
        record = json.loads(record_path.read_text())
        if record['status'] != 'complete' or not record.get('paths'):
            unavailable.append({'id': object_id, 'reason': record.get('error', record['status'])})
            continue
        mesh_path = record_path.parent / record['paths']['mesh']
        expected = record['output_sha256'][record['paths']['mesh']]
        if digest(mesh_path) != expected:
            raise ValueError(f'Native mesh checksum mismatch: {object_id}')
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError(f'Invalid generated mesh: {object_id}')
        fit_height = max(scoring_height(root, spec, 144) for spec in obj['views'])
        score_height = max(eval_size, fit_height)
        views = [load_view(root, spec, score_height) for spec in obj['views']]
        reference_id = record['anchor_frame']
        reference = next(v for v in views if v['frame_id'] == reference_id)
        initial = reference['c2w'] @ np.asarray(record['object_to_camera'], float)
        decompose(initial)
        # Independently verify the official posed asset to catch a double pose.
        posed = trimesh.load(record_path.parent / record['paths']['posed_mesh'], force='mesh', process=False)
        pose_residual = float(np.max(np.abs(transformed(mesh.vertices, np.asarray(record['object_to_camera'])) - posed.vertices)))
        if pose_residual > 2e-5:
            raise ValueError(f'Native raw/posed asset disagreement: {object_id}')
        print(f'Refining {object_id}: {len(mesh.faces)} faces, anchor {reference_id}', flush=True)
        fitting_views = [load_view(root, view['spec'], fit_height) for view in views]
        final, refinement = refine(mesh, initial, fitting_views, iterations, floor=floor['plane_native'])
        partial = trimesh.util.concatenate([observed_mesh(v) for v in views])
        per_view = []
        for view in views:
            per_view.append({'frame_id': view['frame_id'],
                'role': 'anchor; used for pose refinement' if view['frame_id'] == reference_id else 'other supplied view; used for pose refinement',
                'observed_surface': score_view(partial, np.eye(4), view),
                'generated_initial': score_view(mesh, initial, view),
                'generated_refined': score_view(mesh, final, view)})
        comparison = {'object_id': object_id, 'label': obj['label'], 'model': record['model_id'],
            'fitting_grid_height': fit_height, 'evaluation_grid_height': score_height,
            'native_record': str(record_path.relative_to(root)), 'native_record_sha256': digest(record_path),
            'pose_composition_max_abs_residual': pose_residual,
            'initial_object_to_world': initial.tolist(), 'final_object_to_world': final.tolist(),
            'vertices': len(mesh.vertices), 'faces': len(mesh.faces), 'watertight': bool(mesh.is_watertight),
            'generated_appearance': 'native generated vertex colors; unseen surfaces are inferred',
            'views': per_view, 'refinement': refinement}
        write_json(output / f'{object_id}-comparison.json', comparison)
        comparisons.append(comparison)
        assets.append((object_id, mesh, final))
        entries.append({'id': object_id, 'label': obj['label'], 'source': 'generated',
            'model': record['model_id'], 'transform': decompose(final),
            'frame_ids': [v['frame_id'] for v in views],
            'source_inventory_indices': obj.get('source_inventory_indices', []),
            'metrics': {'comparison': f'{object_id}-comparison.json', 'views': per_view,
                        'watertight': bool(mesh.is_watertight), 'faces': len(mesh.faces)}})
    if not assets:
        raise ValueError('No actual generated object assets; refuse to publish a substitute scene')
    floor_view = load_view(root, floor['views'][-1], eval_size)
    floor_mesh = observed_mesh(floor_view)
    assets.append(('observed_floor', floor_mesh, np.eye(4)))
    entries.append({'id': 'observed_floor', 'label': 'Observed workcell floor', 'source': 'observed',
        'model': source.get('geometry', {}).get('model', 'source geometry'), 'transform': decompose(np.eye(4)), 'frame_ids': [floor_view['frame_id']],
        'metrics': {'plane_residual_p95_native': floor['all_point_residual_p95_native']}})
    glb = trimesh.Scene()
    chunks, offset = [], 0
    world_bounds = []
    for entry, (object_id, mesh, transform) in zip(entries, assets):
        visual = mesh.visual.to_color() if mesh.visual.kind == 'texture' else mesh.visual
        colors = visual.vertex_colors[:, :3].astype(np.float32) / 255
        indices = np.asarray(mesh.faces, dtype='<u4').ravel()
        chunk = np.column_stack([mesh.vertices, mesh.vertex_normals, colors]).astype('<f4')
        if not np.isfinite(chunk).all():
            raise ValueError(f'Nonfinite viewer geometry: {object_id}')
        chunks.extend([chunk.tobytes(), indices.tobytes()])
        entry['mesh'] = {'byte_offset': offset, 'vertex_count': len(chunk), 'stride': 9,
                         'index_byte_offset': offset + chunk.nbytes, 'index_count': len(indices), 'index_type': 'uint32'}
        offset += chunk.nbytes + indices.nbytes
        mesh.metadata.update({'object_id': object_id, 'source': entry['source'],
                              'source_inventory_indices': entry.get('source_inventory_indices', [])})
        glb.add_geometry(mesh, geom_name=object_id, node_name=object_id, transform=transform)
        world_bounds.append(transformed(mesh.vertices, transform))
    (output / 'scene.bin').write_bytes(b''.join(chunks))
    glb.export(output / 'scene.glb')
    reloaded = trimesh.load(output / 'scene.glb', force='scene')
    if len(reloaded.geometry) != len(assets) or not np.allclose(reloaded.bounds, glb.bounds, atol=2e-5):
        raise ValueError('Exported GLB lost an object or changed world bounds')
    cameras = []
    (output / 'images').mkdir(exist_ok=True)
    for frame in source['frames']:
        frame_id = frame['frame_id']
        native = root / 'geometry/frames' / frame_id
        image_path = output / 'images' / f'{frame_id}.png'
        shutil.copy2(native / 'canonical.png', image_path)
        width, height = Image.open(image_path).size
        cameras.append({'id': frame_id, 'label': f'Photo {int(frame_id.split("_")[-1])}',
            'width': width, 'height': height, 'K': np.load(native / 'intrinsics.npy').tolist(),
            'camera_to_world': np.load(native / 'camera_to_world.npy').tolist(),
            'image': str(image_path.relative_to(output))})
    all_points = np.concatenate(world_bounds)
    scene = {'version': 1, 'run_id': source['experiment'], 'units': 'uncalibrated native scale (not metres)',
        'up': floor['up_native'], 'bounds': {'min': all_points.min(0).tolist(), 'max': all_points.max(0).tolist()},
        'binary': 'scene.bin', 'glb': 'scene.glb', 'metrics_report': 'metrics.html', 'cameras': cameras, 'objects': entries,
        'limitations': ['Generated hidden surfaces and textures have no photographic ground truth',
            'All supplied views contribute to geometry and placement; scores measure input consistency, not held-out accuracy',
            'Static editable meshes; no robot joints, collision validation or physical calibration'],
        'unavailable_objects': unavailable,
        'object_evidence': '../object-evidence.json' if (root/'object-evidence.json').is_file() else None,
        'floor_plane': floor['plane_native']}
    write_json(output / 'scene.json', scene)
    write_json(output / 'comparisons.json', {'run_id': source['experiment'], 'evaluation_grid_height': eval_size,
        'metrics': {'visible_iou': 'higher is better; foreground-aware silhouette IoU',
            'boundary_error_image_height': 'lower is better; mean bidirectional boundary distance / image height',
            'relative_depth_p50': 'lower is better; median absolute z error / source estimated z on mask overlap'},
        'objects': comparisons, 'unavailable_objects': unavailable})
    write_metrics(output, source['experiment'], comparisons, unavailable)
    print(json.dumps({'output': str(output), 'generated_objects': len(comparisons),
        'unavailable_objects': unavailable, 'glb_bytes': (output / 'scene.glb').stat().st_size}, indent=2), flush=True)


def camera_depth(points, c2w, valid):
    """Row-vector world points -> native OpenCV camera z; invalid means unobserved."""
    points=np.asarray(points);c2w=np.asarray(c2w,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    if points.shape!=valid.shape+(3,) or c2w.shape!=(4,4) or not np.isfinite(c2w).all():
        raise ValueError('Invalid world pointmap / c2w / valid shapes')
    if not np.allclose(c2w[3],[0,0,0,1]) or not np.allclose(c2w[:3,:3].T@c2w[:3,:3],np.eye(3),atol=1e-4):
        raise ValueError('Expected rigid native camera-to-world transform')
    w2c=np.linalg.inv(c2w)
    z=points@w2c[2,:3]+w2c[2,3]
    keep=valid&np.isfinite(points).all(axis=-1)&(np.linalg.norm(points,axis=-1)>1e-6)&np.isfinite(z)&(z>0)
    return np.where(keep,z,0).astype(np.float32),keep
