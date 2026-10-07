"""Place generated object assets in the observed scene and evaluate real views.

This is an experiment, not an implementation of the unreleased GizmoAct policy.
All available object views constrain pose refinement, with each view reported
separately. Native generated assets and their initial poses are kept.
"""

import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import tempfile
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
    from generate_lucida_assets import camera_depth
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
                     '锚定视图 / 参与优化' if view['role'] == 'anchor; used for pose refinement' else '其它视图 / 参与优化']
            for metric in ['visible_iou', 'boundary_error_image_height', 'relative_depth_p50']:
                cells.extend('无重叠' if value is None else f'{value * 100:.2f}%' for value in [before[metric], after[metric]])
            cells.append(f"{view['observed_surface']['visible_iou'] * 100:.2f}%")
            rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
        sections.append(f'<section><h2>{html.escape(comparison["label"])}</h2>'
            f'<p>{comparison["faces"]:,} 三角面 · 封闭网格：{"是" if comparison["watertight"] else "否"} · '
            f'<a href="{html.escape(comparison["object_id"])}-comparison.json">完整记录</a></p>'
            '<div class="scroll"><table><thead><tr><th>输入图</th><th>用途</th>'
            '<th>轮廓 IoU 前 ↑</th><th>后 ↑</th><th>轮廓距离 前 ↓</th><th>后 ↓</th>'
            '<th>相对深度误差 前 ↓</th><th>后 ↓</th><th>观测表面 IoU</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
            f'<img loading="lazy" style="width:100%;height:auto;margin-top:18px" '
            f'src="{html.escape(comparison["object_id"])}-alignment.png" '
            f'alt="{html.escape(comparison["label"])}: input segmentation and before/after projections"></section>')
    missing = (f'<p>其余 {len(unavailable)} 条候选尚未生成。逐项状态见 <a href="comparisons.json">完整记录</a>。</p>') if unavailable else ''
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>工位重建 · 同图对比</title><style>body{font:15px/1.6 system-ui;margin:0;background:#f3f4ef;color:#253b38}main{max-width:1280px;margin:40px auto;padding:0 24px}h1{font-size:30px}h2{font-size:20px}p{max-width:960px}section{background:white;border:1px solid #d3dcd4;border-radius:12px;padding:20px;margin:22px 0}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;text-align:right;border-bottom:1px solid #e0e6e1}td:first-child,th:first-child{text-align:left}a{color:#176c62}</style><main>'
    document += f'<a href="index.html">← 返回可编辑 3D</a><h1>同一批照片，同一物体，同一指标</h1><p>Run：{html.escape(run_id)}</p>'
    document += '<p>“前”是 RecGen 原生生成的网格和预测摆放；“后”是同一网格经过全部可用对象视图共同约束的局部摆放优化。没有替换物体或修改照片。轮廓距离除以图像高度；深度误差是与源几何估计深度比较的中位相对误差，不能解释为实测尺寸精度。</p>'
    document += '<p>“观测表面”只包含照片可见的三角面。各对象使用的视图逐行列出，本表为输入一致性拟合，不是独立测试集指标。封闭网格仅表示几何拓扑，不证明被遮挡的形状正确。</p>'
    document += '<p>这是 RecGen 的实际实验；没有运行未公开的 Lucida GizmoAct 策略。资产为静态可编辑网格，机器人关节和物理碰撞尚未验证。</p>' + missing + ''.join(sections)
    document += '<p><a href="comparisons.json">下载所有指标</a> · <a href="scene.glb">下载 GLB</a></p></main></html>'
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


def add_scene_context(root, output, ehs_repo, *, scene=None, source=None):
    """Export every bound observation against shared native surfaces, without fitting."""
    sys.path.insert(0,str(Path(ehs_repo).resolve()))
    from ehs_spatial.object_evidence import load_candidate_mask
    from ehs_spatial.measurements import measure_observed_points
    from ehs_spatial.providers.map_anything import input_mask_to_canonical
    root, output = Path(root).resolve(), Path(output).resolve()
    observed_only = scene is not None
    scene = scene if observed_only else json.loads((output/'scene.json').read_text())
    source = source if source is not None else json.loads((root/'manifest.json').read_text())
    registry = json.loads((root/'object-evidence.json').read_text())
    frames = {f['frame_id']:f for f in source['frames']}
    measurement_scene = {}
    scene_path = root/'scene.json'
    if scene_path.is_file():
        if registry['source_sha256'].get('scene.json') != digest(scene_path):
            raise ValueError('Saved floor/scale reference differs from the full registry')
        measurement_scene = json.loads(scene_path.read_text())
    plane=np.asarray(measurement_scene.get('floor_plane'),float)
    if plane.shape==(4,) and np.isfinite(plane).all() and np.linalg.norm(plane[:3])>1e-9:
        scene.update(up=(plane[:3]/np.linalg.norm(plane[:3])).tolist(),floor_plane=plane.tolist(),
            up_reference='saved_native_floor',floor_reference={'source_sha256':registry['source_sha256']['scene.json']})
    calibration = registry.get('measurement_reference',{}).get('calibration')
    if calibration:
        bound = [path for path,sha in registry['source_sha256'].items() if sha==calibration.get('source_sha256')]
        if not bound or any(digest(root/path)!=calibration['source_sha256'] for path in bound):
            raise ValueError('Explicit camera-height evidence is missing or changed')
    def measurements(candidate, points, mask_pixels):
        fid=candidate.get('frame_id')
        return measure_observed_points(points,measurement_scene,mask_pixels=mask_pixels,calibration=calibration,
            source={'frame_id':fid,'candidate_id':candidate['id'],'scene_sha256':registry['source_sha256'].get('scene.json'),
                'image_sha256':frames.get(fid,{}).get('sha256'),'mask_sha256':candidate['mask'].get('sha256'),
                'pointmap_sha256':registry['source_sha256'].get(f'geometry/frames/{fid}/pts3d.npy')})
    objects = ({o['object_id']:o for o in json.loads((root/'evidence/objects.json').read_text())['objects']}
               if any(o['source']=='generated' for o in scene['objects']) else {})
    frozen = {p.name:digest(p) for p in output.iterdir() if p.name.endswith(('-comparison.json','-alignment.png'))
              or p.name in {'metrics.html','comparisons.json'}}
    previous_contexts = [o['id'] for o in scene['objects'] if o.get('role')=='context']
    existing = [o for o in scene['objects'] if o.get('role')!='context']
    scene['objects'] = existing
    raw_binary = (output/scene['binary']).read_bytes() if scene.get('binary') else None
    chunks, offset = [], 0
    for entry in existing:
        old = entry['mesh']
        if raw_binary is None:
            raw = gzip.decompress((output/old['asset']['path']).read_bytes())
            if hashlib.sha256(raw).hexdigest() != old['asset']['sha256']:
                raise ValueError('Existing packed mesh changed: '+entry['id'])
        else:
            raw = raw_binary[old['byte_offset']:old['index_byte_offset']+old['index_count']*4]
        if len(raw) != old['vertex_count']*36+old['index_count']*4:
            raise ValueError('Existing mesh byte contract changed: '+entry['id'])
        chunks.append(raw)
        entry['mesh'] = {k:v for k,v in old.items() if k != 'asset'}
        entry['mesh'].update(byte_offset=offset,index_byte_offset=offset+old['vertex_count']*36)
        offset += len(raw)
    glb = trimesh.load(output/'scene.glb',force='scene',process=False) if existing else trimesh.Scene()
    glb.delete_geometry(previous_contexts)
    contexts, frame_errors = {}, {}
    for fid,frame in frames.items():
        relative=Path('geometry/frames')/fid
        spec={'frame_id':fid,**{key:(relative/name).as_posix() for key,name in
            [('canonical_rgb_path','canonical.png'),('pointmap_path','pts3d.npy'),('valid_path','valid_mask.npy'),
             ('content_valid_path','content_valid_mask.npy'),('conf_path','conf.npy'),('K_path','intrinsics.npy'),
             ('c2w_path','camera_to_world.npy')]}}
        if observed_only:
            spec.pop('content_valid_path')
            spec['content_rect_xyxy']=frame.get('content_rect_xyxy')
        try:
            for key,path in spec.items():
                if key in {'frame_id','content_valid_path','content_rect_xyxy'}:continue
                if registry['source_sha256'].get(path) != digest(root/path):
                    raise ValueError('Native context source differs from full registry: '+path)
            provider=(Path('geometry/provider')/(fid+'.json')).as_posix()
            if provider in registry['source_sha256'] and digest(root/provider)!=registry['source_sha256'][provider]:
                raise ValueError('Original-mask pixel mapping changed: '+fid)
            if observed_only:
                original=(root/frame['input']).resolve()
                if not original.is_relative_to(root) or digest(original)!=frame['sha256']:
                    raise ValueError('Source photograph is missing or changed')
                affine=np.asarray(frame.get('input_to_canonical_pixel_centres'),float)
                if (affine.shape!=(3,3) or not np.isfinite(affine).all()
                        or not np.allclose(affine[2],[0,0,1]) or abs(np.linalg.det(affine))<1e-12):
                    raise ValueError('No invertible recorded source-photo pixel mapping')
                with Image.open(original) as photo:
                    if photo.size!=(frame['width'],frame['height']):
                        raise ValueError('Source photograph dimensions differ from the registry')
            view=native_view(root,spec)
            view['valid'] &= np.linalg.norm(view['points'],axis=-1)>1e-6
            contexts[fid]={'view':view,'exclude':np.zeros_like(view['valid']),'objects':[],'bindings':[]}
        except (OSError,ValueError,KeyError,TypeError) as error:
            if not observed_only:raise
            frame_errors[fid]=str(error).replace(str(root),'<source run>')
    for entry in existing:
        if entry['source'] != 'generated':continue
        oid = entry['id'];obj = objects[oid]
        record_path = root/'generation'/oid/'output.json'
        record = json.loads(record_path.read_text())
        comparison = json.loads((output/entry['metrics']['comparison']).read_text())
        if digest(record_path) != comparison['native_record_sha256']:
            raise ValueError('Generated result is not bound to this native record: '+oid)
        fid = record['anchor_frame'];frame = frames[fid]
        spec = next(v for v in obj['views'] if v['frame_id'] == fid)
        bound = next(v for v in record['source_views'] if v['frame_id'] == fid)
        for key in ['canonical_rgb_path','canonical_mask_path','pointmap_path','content_valid_path','conf_path','K_path','c2w_path']:
            if digest(root/spec[key]) != bound['sha256'][key]:
                raise ValueError('Anchor context differs from generated source evidence: '+key)
        if bound['source_input_sha256'] != frame['sha256']:
            raise ValueError('Anchor context source photograph differs from native generation')
        if not np.array_equal(np.asarray(bound['input_to_canonical_pixel_centres']),
                              np.asarray(frame['input_to_canonical_pixel_centres'])):
            raise ValueError('Anchor source-photo affine differs from native generation')
        context = contexts[fid]
        context['exclude'] |= np.load(root/spec['canonical_mask_path'],allow_pickle=False).astype(bool)
        context['objects'].append(oid)
        context['bindings'].append({'object_id':oid,'native_record_sha256':digest(record_path),
                                    'canonical_mask_sha256':bound['sha256']['canonical_mask_path']})
        entry['context_id'] = 'observed_context_'+fid
        entry['reference_frame'] = fid
    for entry in existing:
        if entry['id'] == 'observed_floor':
            entry.update(selectable=False,editable=False)
    added = []
    for fid,context in list(contexts.items()):
        view = context['view'];mask = view['valid'] & ~context['exclude']
        try:
            mesh,pixel_faces = observed_mesh(dict(view,mask=mask),return_pixel_faces=True)
        except ValueError as error:
            if not observed_only:raise
            frame_errors[fid]=str(error);del contexts[fid];continue
        context['pixel_faces']=pixel_faces
        oid = 'observed_context_'+fid
        raw, layout = mesh_bytes(mesh)
        layout.update(byte_offset=offset,index_byte_offset=offset+layout['index_byte_offset'])
        offset += len(raw);chunks.append(raw)
        provenance = {'source_run_id':source.get('source_run_id',source['experiment']), 'frame_id':fid,
            'source_image_sha256':frames[fid]['sha256'],'geometry_sha256':{key:digest(root/view['spec'][key])
                for key in ['pointmap_path','content_valid_path','conf_path','K_path','c2w_path','canonical_rgb_path'] if key in view['spec']},
            'mask_rule':'valid observed source-frame pixels excluding exact generated-object masks; no multi-frame fusion',
            'excluded_objects':context['bindings'],'observed_support_pixels':int(mask.sum()),
            'triangulation':'existing observed_mesh adjacent pixels and depth-relative edge gate; no filling',
            'metric_scale_known':False}
        if observed_only:
            provenance['content_rect_xyxy']=frames[fid]['content_rect_xyxy']
        Image.fromarray(mask.astype(np.uint8)*255).save(output/(oid+'-mask.png'))
        provenance['canonical_mask_path'] = oid+'-mask.png'
        provenance['canonical_mask_sha256'] = digest(output/provenance['canonical_mask_path'])
        entry = {'id':oid,'label':'同机位观测工位背景','source':'observed','role':'context','editable':False,'selectable':False,
            'model':source.get('geometry',{}).get('model','source geometry'),'frame_ids':[fid],
            'context_for':context['objects'],'transform':decompose(np.eye(4)),'mesh':layout,'provenance':provenance}
        added.append(entry);glb.add_geometry(mesh,geom_name=oid,node_name=oid,transform=np.eye(4))
    scene['objects'].extend(added)
    if observed_only:
        if scene['up'] is None and contexts:
            scene['up']=(-next(iter(contexts.values()))['view']['c2w'][:3,1]).tolist()
            scene['up_reference']='camera_up_navigation_only'
        for fid,context in contexts.items():
            view=context['view'];target=output/'images'/(fid+'.png')
            shutil.copyfile(root/view['spec']['canonical_rgb_path'],target)
            scene['cameras'].append({'id':fid,'label':fid,'width':view['rgb'].shape[1],
                'height':view['rgb'].shape[0],'K':view['K'].tolist(),'camera_to_world':view['c2w'].tolist(),
                'image':target.relative_to(output).as_posix()})
    regions,unavailable=[],[]
    generated={o['id'] for o in existing if o['source']=='generated'}
    up=np.asarray(scene['up'],float) if scene['up'] is not None else None
    if up is not None:up/=np.linalg.norm(up)
    region_dir=output/'regions';region_dir.mkdir(exist_ok=True)
    for candidate in registry['candidates']:
        oid=candidate['id'];fid=candidate['frame_id']
        if oid in generated:
            # Immutable observed evidence belongs to the source, not to edits of the generated transform.
            entry=next(o for o in existing if o['id']==oid)
            view=contexts[fid]['view']
            mask=input_mask_to_canonical(load_candidate_mask(root,candidate),root,fid,view['valid'].shape)
            entry['measurements']=measurements(candidate,view['points'][mask & view['valid']],int(mask.sum()))
            continue
        item={'id':oid,'label':candidate['label'],'labels':candidate.get('labels',[]),
              'reference_frame':fid,'frame_ids':[fid] if fid else [],'source':'observed',
              'source_inventory_indices':candidate.get('inventory_indices',[]),
              'editable':False,'selectable':True,'context_id':'observed_context_'+fid if fid else None,
              'source_bbox':{'bbox_xyxy':candidate['mask'].get('bbox'),
                             'resolution':candidate['mask'].get('resolution'),
                             'shape_hw':candidate['mask'].get('shape_hw')},
              'provenance':{'source_run_id':registry['run_id'],'candidate_id':oid,
                  'source_image_sha256':frames[fid]['sha256'] if fid in frames else None,
                  'mask_sha256':candidate['mask'].get('sha256'),'source_refs':candidate['source_refs'],
                  'physical_identity':'single-frame mask observation; overlapping labels and cross-frame identities are not merged',
                  'axes':'native world axes; not an inferred physical object orientation','metric_scale_known':False}}
        item['measurements']=measurements(candidate,np.empty((0,3)),0)
        try:
            if fid not in contexts or candidate['mask']['status']!='available':
                reason=frame_errors.get(fid) or candidate['mask'].get('reason')
                raise ValueError(reason if reason not in {None,'','None'} else 'No source frame and usable mask are recorded')
            context=contexts[fid];view=context['view']
            mask=input_mask_to_canonical(load_candidate_mask(root,candidate),root,fid,view['valid'].shape)
            supported=mask & view['valid'];points=view['points'][supported]
            item['measurements']=measurements(candidate,points,int(mask.sum()))
            item['provenance']['supported_points']=len(points)
            if not len(points):raise ValueError('No finite, valid, positive-depth native points support this mask')
            item['bounds_native']={'min':points.min(0).tolist(),'max':points.max(0).tolist()}
            height=points@up
            item['ground_extent_native']={'min':float(height.min()),'max':float(height.max())}
            image_path=region_dir/(oid+'.mask.png');Image.fromarray(mask.astype(np.uint8)*255).save(image_path)
            y,x=np.nonzero(mask)
            item['mask']={'path':image_path.relative_to(output).as_posix(),'sha256':digest(image_path),'pixels':int(mask.sum()),
                          'bbox_xyxy':[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1],
                          'shape_hw':list(mask.shape),'resolution':'canonical'}
            face_ids=np.flatnonzero(mask.ravel()[context['pixel_faces']].all(1)).astype('<u4')
            if not len(face_ids):
                if len(points)>=8 and (mask & context['exclude']).any():
                    # Prove connected surface support before replacement-mask exclusion.
                    # No extra mesh is exported and no object identity is inferred.
                    observed_mesh(dict(view,mask=supported))
                    item.update(faces=None,surface_status='covered_by_generated',
                        reason='All connected observed triangles for this mask are excluded from the background by generated replacement masks; native bounds remain available')
                    regions.append(item)
                    continue
                raise ValueError('No connected observed triangles remain for this mask after invalid-depth and generated-mask exclusion')
            path=region_dir/(oid+'.faces.bin.gz');raw=face_ids.tobytes();packed=gzip.compress(raw,mtime=0);path.write_bytes(packed)
            item.update(surface_status='observed',faces={'count':len(face_ids),'encoding':'uint32-triangle-indices',
                'asset':{'path':path.relative_to(output).as_posix(),'bytes':len(raw),'packed_bytes':len(packed),
                         'sha256':hashlib.sha256(raw).hexdigest()}})
            regions.append(item)
        except (OSError,ValueError,KeyError,IndexError,TypeError) as error:
            item.update(selectable=False,reason=str(error).replace(str(root),'<source run>'),status='unavailable')
            unavailable.append(item)
    scene['observed_regions']=regions
    scene['unavailable_regions']=unavailable
    scene['observed_regions_summary']={'registry_candidates':len(registry['candidates']),
        'generated_exact_candidates':len(generated),'observed_regions':len(regions),'unavailable_regions':len(unavailable),
        'regions_with_surface':sum(r['faces'] is not None for r in regions),
        'regions_with_bounds_only':sum(r['faces'] is None for r in regions),
        'counting_unit':'source-frame mask observations, not unique physical objects',
        'overlaps':'full memberships retained; no triangle ownership partition or identity merge',
        'registry_sha256':digest(root/'object-evidence.json')}
    assert len(regions)+len(unavailable)+len(generated)==len(registry['candidates'])
    for camera in scene['cameras']:
        fid=camera['id'];frame=frames[fid];original=(root/frame['input']).resolve()
        if not original.is_relative_to(root) or digest(original) != frame['sha256']:
            raise ValueError('Source photograph is missing or changed: '+fid)
        geom=root/'geometry/frames'/fid
        K=np.load(geom/'intrinsics.npy',allow_pickle=False)
        C=np.load(geom/'camera_to_world.npy',allow_pickle=False)
        if not np.array_equal(K,np.asarray(camera['K'])) or not np.array_equal(C,np.asarray(camera['camera_to_world'])):
            raise ValueError('Existing scene camera differs from native geometry: '+fid)
        A=np.asarray(frame['input_to_canonical_pixel_centres'],float)
        if A.shape!=(3,3) or not np.isfinite(A).all() or not np.allclose(A[2],[0,0,1]):
            raise ValueError('Source full-photo pixel affine is invalid: '+fid)
        with Image.open(original) as image:
            if image.size != (frame['width'],frame['height']):
                raise ValueError('Source photo dimensions disagree with recorded affine: '+fid)
            orientation=image.getexif().get(274,1)
            suffix={"JPEG":".jpg","PNG":".png","WEBP":".webp"}[image.format]
            target=output/'images'/(fid+'-original'+(suffix if orientation==1 else '.png'))
            if orientation==1:
                shutil.copyfile(original,target);encoding='unchanged source bytes'
            else:
                # The generator uses PIL.convert(RGB), without EXIF rotation.
                # Encode that exact pixel grid so browsers cannot rotate it again.
                image.convert('RGB').save(target);encoding='same decoded RGB grid as generator, PNG without EXIF orientation'
        camera.update(context_id='observed_context_'+fid,original_image=target.relative_to(output).as_posix(),original_width=frame['width'],original_height=frame['height'],
            original_K=(np.linalg.inv(A)@K).tolist(),original_sha256=digest(target),original_source_sha256=frame['sha256'],
            input_to_canonical_pixel_centres=A.tolist(),original_encoding=encoding)
    if scene['objects']:
        (output/'scene.bin').write_bytes(b''.join(chunks))
        glb.export(output/'scene.glb')
        reloaded=trimesh.load(output/'scene.glb',force='scene',process=False)
        if len(reloaded.geometry)!=len(scene['objects']) or not np.allclose(reloaded.bounds,glb.bounds,atol=2e-5):
            raise ValueError('Context GLB export changed object count or coordinates')
        scene.update(binary='scene.bin',glb='scene.glb',bounds={'min':glb.bounds[0].tolist(),'max':glb.bounds[1].tolist()})
    else:
        scene.update(bounds=None,reason='No frame has a valid connected native surface and verified photo mapping')
    scene.pop('source_binary',None);scene.pop('binary_asset',None)
    note='Each context is a partial observed surface from one source frame; unseen regions remain absent and capture states are not fused. Region masks can overlap and do not establish unique physical identities.'
    if note not in scene['limitations']:scene['limitations'].append(note)
    write_json(output/'scene.json',scene)
    assert all(digest(output/name)==sha for name,sha in frozen.items()), 'Frozen comparison artifacts changed'
    validation={'status':'ready' if scene['objects'] else 'unavailable','frame_errors':frame_errors,
        'new_model_calls':0,'pose_optimization_calls':0,'source_run':source.get('source_run_id',source['experiment']),
        'frozen_metrics_sha256':frozen,'contexts':[{'id':o['id'],'frame_id':o['frame_ids'][0],
            'triangles':o['mesh']['index_count']//3,'source_image_sha256':o['provenance']['source_image_sha256']} for o in added],
        'full_source_cameras':[c['id'] for c in scene['cameras']],
        'observed_regions_summary':scene['observed_regions_summary']}
    write_json(output/'context-validation.json',validation)
    return validation


def context_revision(root, source_result, output, ehs_repo):
    """Copy a frozen result and augment only its scene assets; never rerun pose fitting."""
    root,source_result,output=map(lambda p:Path(p).resolve(),[root,source_result,output])
    if output.exists() or output.is_relative_to(source_result) or source_result.is_relative_to(output):
        raise ValueError('Use a new context revision directory outside the frozen result')
    shutil.copytree(source_result,output)
    return add_scene_context(root,output,ehs_repo)


def assemble_observed_scene(source_run, output, ehs_repo):
    """Build a normal product run's native observation scene, with zero generation."""
    root,output=Path(source_run).resolve(),Path(output).resolve()
    if output.exists() or root.is_relative_to(output):
        raise ValueError('Observed output must be a new directory, not an ancestor of the source run')
    registry=json.loads((root/'object-evidence.json').read_text())
    frames=[]
    for frame in registry['frames']:
        fid=frame['frame_id']
        if not isinstance(fid,str) or Path(fid).name!=fid or fid.startswith('.'):
            raise ValueError('Unsafe source frame ID')
        frames.append({'frame_id':fid,'input':frame['image_path'],'sha256':frame['source_sha256'],
            'width':frame['width'],'height':frame['height'],
            'input_to_canonical_pixel_centres':frame.get('input_to_canonical_pixel_centres'),
            'content_rect_xyxy':frame.get('content_rect_xyxy')})
    if len({f['frame_id'] for f in frames})!=len(frames):
        raise ValueError('Duplicate source frame ID')
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').is_file() else {}
    source={'experiment':registry['run_id'],'source_run_id':registry['run_id'],'frames':frames,
            'geometry':{'model':manifest.get('providers',{}).get('map_anything_model_id','source geometry')}}
    output.mkdir(parents=True);(output/'images').mkdir()
    scene={'version':1,'run_id':registry['run_id'],'label':'观测工位 / Observed workcell',
        'units':'native reconstruction units; scale provenance is recorded in each measurement',
        'up':None,'up_reference':None,'floor_plane':None,'cameras':[],'objects':[],
        'limitations':['Native visible surfaces only; no object completion, generative model or pose optimization is used',
            'Object regions are per-frame mask observations; physical identity is not inferred across labels or viewpoints',
            'Dimensions describe observed support, with the saved scale provenance; hidden object dimensions remain unknown']}
    validation=add_scene_context(root,output,ehs_repo,scene=scene,source=source)
    write_json(output/'observed-validation.json',validation)
    return validation


def observed_self_check(ehs_repo):
    """Exercise normal-run frames, missing floor and empty geometry without a model."""
    with tempfile.TemporaryDirectory(prefix='observed-scene-check-') as temporary:
        root=Path(temporary)/'source';root.mkdir()
        h,w=8,10;yy,xx=np.indices((h,w));K=np.array([[50.,0,4.5],[0,50.,3.5],[0,0,1]])
        points=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T*3
        frames=[];candidates=[]
        for number in [1,2]:
            fid=f'frame_{number:04d}';native=root/'geometry/frames'/fid;native.mkdir(parents=True)
            original=root/f'image_{number}.png';Image.new('RGB',(w*2,h*2),'gray').save(original)
            Image.new('RGB',(w,h),'gray').save(native/'canonical.png')
            for name,array in [('pts3d',points if number==1 else points*0),('valid_mask',np.ones((h,w),bool)),
                               ('conf',np.ones((h,w))),('intrinsics',K),('camera_to_world',np.eye(4))]:
                np.save(native/(name+'.npy'),array)
            mask=np.zeros((h,w),bool);mask[1:6,1:7]=True
            path=root/(fid+'-mask.png');Image.fromarray(mask.astype(np.uint8)*255).save(path)
            frames.append({'frame_id':fid,'image_path':original.name,'source_sha256':digest(original),
                'width':w*2,'height':h*2,'input_to_canonical_pixel_centres':[[.5,0,-.25],[0,.5,-.25],[0,0,1]],
                'content_rect_xyxy':[0,0,w,h]})
            candidates.append({'id':fid+'-region','label':'synthetic observation','frame_id':fid,'source_refs':[],
                'mask':{'status':'available','resolution':'canonical','shape_hw':[h,w],'bbox':[1,1,7,6],
                    'ref':{'path':path.name,'sha256':digest(path),'encoding':'png','shape_hw':[h,w]}}})
        candidates.append({'id':'unlocalized','label':'unlocalized','frame_id':None,'source_refs':[],
                           'mask':{'status':'unavailable','reason':'No source mask'}})
        write_json(root/'scene.json',{'floor_plane':[0,-1,0,1.5],'scale_source':'moge_anchor','scale_factor':2.0})
        registry={'run_id':'source','frames':frames,'candidates':candidates}
        def save_registry():
            registry['source_sha256']={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*')
                                      if p.is_file() and p.name!='object-evidence.json'}
            write_json(root/'object-evidence.json',registry)
        save_registry()
        result=Path(temporary)/'complete'
        validation=assemble_observed_scene(root,result,ehs_repo)
        scene=json.loads((result/'scene.json').read_text())
        assert len(scene['objects'])==1 and len(scene['observed_regions'])==1 and len(scene['unavailable_regions'])==2
        assert validation['new_model_calls']==validation['pose_optimization_calls']==0
        assert scene['up_reference']=='saved_native_floor' and scene['up']==[0.,-1.,0.]
        assert scene['observed_regions'][0]['measurements']['status']=='available'
        assert np.allclose(np.asarray(frames[0]['input_to_canonical_pixel_centres'])@scene['cameras'][0]['original_K'],K)
        assert not (root/'generation').exists() and not (root/'evidence').exists()
        assert registry['source_sha256']=={p:digest(root/p) for p in registry['source_sha256']}
        # A camera is a navigation reference, never a fabricated floor measurement.
        (root/'scene.json').unlink();save_registry()
        no_floor=Path(temporary)/'no-floor';assemble_observed_scene(root,no_floor,ehs_repo)
        scene=json.loads((no_floor/'scene.json').read_text())
        assert scene['up_reference']=='camera_up_navigation_only' and scene['floor_plane'] is None
        assert scene['observed_regions'][0]['measurements']['status']=='unavailable'
        np.save(root/'geometry/frames/frame_0001/pts3d.npy',points*0);save_registry()
        empty=Path(temporary)/'empty';validation=assemble_observed_scene(root,empty,ehs_repo)
        scene=json.loads((empty/'scene.json').read_text())
        assert validation['status']=='unavailable' and len(scene['unavailable_regions'])==3
        assert scene['objects']==[] and scene['bounds'] is None
        assert not (empty/'scene.bin').exists() and not (empty/'scene.glb').exists()
    cube = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)]) * [1, 2, 1]  # 2 units tall, floor y = 0
    assert floor_penalty(cube, np.array([0., 1., 0., 0.]))[0] == 0.0, 'resting on the floor is free'
    assert floor_penalty(cube + [0, 5, 0], np.array([0., 1., 0., 0.]))[0] == 0.0, 'floating is not penalised'
    assert abs(floor_penalty(cube - [0, .2, 0], np.array([0., 1., 0., 0.]))[0] - FLOOR_CONTACT_WEIGHT * .1) < 1e-9, 'sunk 10 % of height'
    print('PASS: observed-only shared core; partial frames, source cameras, real floor, unknown floor and empty geometry; zero model calls; floor hinge')


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
            'generated_appearance': 'native RecGen vertex colors; unseen surfaces are inferred',
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
    entries.append({'id': 'observed_floor', 'label': '观测到的工位地面', 'source': 'observed',
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
        cameras.append({'id': frame_id, 'label': f'照片 {int(frame_id.split("_")[-1])}',
            'width': width, 'height': height, 'K': np.load(native / 'intrinsics.npy').tolist(),
            'camera_to_world': np.load(native / 'camera_to_world.npy').tolist(),
            'image': str(image_path.relative_to(output))})
    all_points = np.concatenate(world_bounds)
    scene = {'version': 1, 'run_id': source['experiment'], 'units': '未标定尺度（非米）',
        'up': floor['up_native'], 'bounds': {'min': all_points.min(0).tolist(), 'max': all_points.max(0).tolist()},
        'binary': 'scene.bin', 'glb': 'scene.glb', 'metrics_report': 'metrics.html', 'cameras': cameras, 'objects': entries,
        'limitations': ['RecGen experiment, not a reproduction of the unreleased Lucida GizmoAct policy',
            'Generated hidden surfaces and textures have no photographic ground truth',
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
    shutil.copy2(Path(__file__).with_name('lucida_viewer.html'), output / 'index.html')
    print(json.dumps({'output': str(output), 'generated_objects': len(comparisons),
        'unavailable_objects': unavailable, 'glb_bytes': (output / 'scene.glb').stat().st_size}, indent=2), flush=True)


def self_check(ehs_repo):
    mesh = trimesh.creation.box([1, 1, 1])
    pose = np.eye(4)
    pose[2, 3] = 3
    assert np.allclose(matrix(decompose(pose)), pose)
    yy, xx = np.indices((144, 144))
    rays = np.concatenate([np.zeros((144, 144, 3)), np.stack([(xx-71.5)/150, (yy-71.5)/150, np.ones_like(xx)], -1)], -1).astype(np.float32)
    depth = cast_depth(mesh, pose, rays)
    mask = np.isfinite(depth)
    view = {'rays': rays, 'depth': np.where(mask, depth, 0), 'target': mask, 'frame_id': 'synthetic'}
    exact = score_view(mesh, pose, view)
    assert exact['visible_iou'] == 1 and exact['relative_depth_p95'] == 0
    shifted = pose.copy(); shifted[0, 3] += .2
    assert score_view(mesh, shifted, view)['loss'] > exact['loss']
    reflected = pose.copy(); reflected[0, 0] = -1
    try:
        decompose(reflected)
    except ValueError:
        pass
    else:
        raise AssertionError('Reflection must not be silently interpreted as rotation')
    stretched = pose.copy()
    stretched[:3, :3] = Rotation.from_euler('xyz', [.1, -.05, .15]).as_matrix() @ np.diag([.8, 1.2, .9])
    independent = o3d.t.geometry.RaycastingScene()
    independent.add_triangles(o3d.core.Tensor(transformed(mesh.vertices, stretched).astype(np.float32)),
                              o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    world_hits = independent.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    local_hits = cast_depth(mesh, stretched, rays)
    assert np.array_equal(np.isfinite(world_hits), np.isfinite(local_hits))
    assert np.allclose(world_hits[np.isfinite(world_hits)], local_hits[np.isfinite(local_hits)], atol=1e-5)
    final, _ = refine(mesh, stretched, [view], max_iterations=4)
    assert np.allclose(matrix(decompose(final)), final)
    # Exercise the actual file/export contract in a disposable synthetic run.
    # Never publish this fixture in the user's experiment directory.
    with tempfile.TemporaryDirectory(prefix='lucida-assembly-check-') as directory:
        root = Path(directory)
        native = root / 'geometry/frames/frame_0001'
        generation = root / 'generation/cube'
        for path in [native, generation, root / 'evidence']:
            path.mkdir(parents=True)
        foreground = depth.copy()
        floor_z = np.full_like(depth, np.inf)
        down = rays[..., 4] > 0
        floor_z[down] = 1.5 / rays[..., 4][down]
        floor_mask = (floor_z < depth) & (floor_z < 30)
        foreground[floor_mask] = floor_z[floor_mask]
        valid = np.isfinite(foreground)
        points = rays[..., 3:] * np.where(valid, foreground, 0)[..., None]
        K = np.array([[150., 0, 71.5], [0, 150., 71.5], [0, 0, 1.]])
        Image.new('RGB', (144, 144), '#608088').save(native / 'canonical.png')
        for name, array in [('pts3d', points), ('valid_mask', valid), ('content_valid_mask', np.ones_like(valid)),
                            ('conf', valid.astype(float)), ('intrinsics', K), ('camera_to_world', np.eye(4))]:
            np.save(native / f'{name}.npy', array)
        object_mask = mask & ~floor_mask
        np.save(root / 'evidence/cube_mask.npy', object_mask)
        np.save(root / 'evidence/floor_mask.npy', floor_mask)
        spec = {'frame_id': 'frame_0001', 'canonical_mask_path': 'evidence/cube_mask.npy',
                'canonical_rgb_path': 'geometry/frames/frame_0001/canonical.png',
                'pointmap_path': 'geometry/frames/frame_0001/pts3d.npy',
                'K_path': 'geometry/frames/frame_0001/intrinsics.npy',
                'c2w_path': 'geometry/frames/frame_0001/camera_to_world.npy',
                'valid_path': 'geometry/frames/frame_0001/valid_mask.npy',
                'content_valid_path': 'geometry/frames/frame_0001/content_valid_mask.npy',
                'conf_path': 'geometry/frames/frame_0001/conf.npy'}
        assert scoring_height(root, spec, 48) == 48
        small = np.zeros_like(object_mask)
        small[70:77, 70:77] = True
        np.save(root / 'evidence/small.npy', small)
        small_spec = dict(spec, canonical_mask_path='evidence/small.npy')
        assert scoring_height(root, small_spec, 24) > 24
        small[:] = False
        small[70, 70:77] = True
        np.save(root / 'evidence/small.npy', small)
        try:
            scoring_height(root, small_spec, 24)
        except ValueError:
            pass
        else:
            raise AssertionError('Upsampling must not invent eight independent depth samples')
        # A non-square source with EXIF rotation exercises the raw decoded grid
        # used by native inference, rather than browser-dependent orientation.
        source_image = root / 'source.jpg'
        source_rgb = np.zeros((216,288,3),dtype=np.uint8)
        source_rgb[...,0] = np.arange(288,dtype=np.uint16)[None,:] % 256
        source_rgb[...,1] = np.arange(216,dtype=np.uint8)[:,None]
        exif=Image.Exif();exif[274]=6
        Image.fromarray(source_rgb).save(source_image,exif=exif)
        affine = [[.5,0,-.25],[0,2/3,-1/6],[0,0,1]]
        frame = {'frame_id':'frame_0001','input':'source.jpg','sha256':digest(source_image),
                 'width':288,'height':216,'input_to_canonical_pixel_centres':affine}
        manifest = {'experiment':'synthetic-self-check','frames':[frame]}
        write_json(root / 'manifest.json', manifest)
        write_json(root / 'evidence/objects.json', {'objects': [{'object_id': 'cube', 'label': 'Synthetic cube', 'views': [spec]}]})
        # Two overlapping source observations keep both memberships, without
        # duplicating surfaces or inferring that their labels identify one object.
        candidates=[]
        for oid,label,mask_value in [('cube','Synthetic cube',object_mask),('floor-region','Floor region',floor_mask),
                                     ('overlap-region','Overlapping observation',floor_mask & (xx<100)),
                                     ('covered-region','Separate covered observation',object_mask)]:
            mask_path=root/'evidence'/(oid+'.png')
            Image.fromarray(mask_value.astype(np.uint8)*255).save(mask_path)
            candidates.append({'id':oid,'label':label,'frame_id':'frame_0001','source_refs':[],
                'mask':{'status':'available','resolution':'canonical','shape_hw':[144,144],
                    'bbox':[0,0,144,144],'ref':{'path':mask_path.relative_to(root).as_posix(),
                    'sha256':digest(mask_path),'encoding':'png','shape_hw':[144,144]}}})
        candidates.append({'id':'unlocalized','label':'Unlocalized claim','frame_id':None,'source_refs':[],
                           'mask':{'status':'unavailable','reason':'No source mask'}})
        write_json(root/'object-evidence.json',{'run_id':'synthetic-self-check','candidates':candidates,
            'source_sha256':{p.relative_to(root).as_posix():digest(p) for p in native.iterdir() if p.name!='content_valid_mask.npy'}})
        write_json(root / 'evidence/floor.json', {'views': [{**spec, 'canonical_mask_path': 'evidence/floor_mask.npy'}],
                   'up_native': [0, -1, 0], 'plane_native': [0, -1, 0, 1.5], 'all_point_residual_p95_native': 0})
        mesh.export(generation / 'object.ply')
        mesh.copy().apply_transform(pose).export(generation / 'posed-object.ply')
        write_json(generation / 'output.json', {'status': 'complete', 'anchor_frame': 'frame_0001',
            'model_id': 'synthetic-fixture', 'object_to_camera': pose.tolist(),
            'paths': {'mesh': 'object.ply', 'posed_mesh': 'posed-object.ply'},
            'output_sha256': {'object.ply': digest(generation / 'object.ply')},
            'source_views':[{'frame_id':'frame_0001','source_input_sha256':frame['sha256'],
                'input_to_canonical_pixel_centres':affine,'sha256':{k:digest(root/spec[k]) for k in
                ['canonical_rgb_path','canonical_mask_path','pointmap_path','content_valid_path','conf_path','K_path','c2w_path']}}]})
        assemble(root, iterations=2, eval_size=48)
        result = json.loads((root / 'result/scene.json').read_text())
        assert len(result['objects']) == 2
        expected_bytes = sum(x['mesh']['vertex_count'] * 36 + x['mesh']['index_count'] * 4 for x in result['objects'])
        assert (root / 'result/scene.bin').stat().st_size == expected_bytes
        assert (root / 'result/metrics.html').exists()
        before = {p.relative_to(root/'result').as_posix():digest(p) for p in (root/'result').rglob('*') if p.is_file()}
        validation=context_revision(root,root/'result',root/'context-result',ehs_repo)
        contextual=json.loads((root/'context-result/scene.json').read_text())
        assert len(contextual['objects'])==3 and validation['new_model_calls']==0
        assert (root/'context-result/scene.bin').read_bytes()[:expected_bytes] == (root/'result/scene.bin').read_bytes()
        assert all(x['transform']==y['transform'] and x.get('metrics')==y.get('metrics')
                   for x,y in zip(result['objects'],contextual['objects']))
        full=contextual['cameras'][0]
        assert (full['original_width'],full['original_height'])==(288,216)
        assert np.allclose(np.asarray(affine)@np.asarray(full['original_K']),K,atol=1e-12)
        assert np.array_equal(np.asarray(Image.open(root/'context-result'/full['original_image'])),
                              np.asarray(Image.open(source_image).convert('RGB')))
        context_mask=np.asarray(Image.open(root/'context-result'/contextual['objects'][-1]['provenance']['canonical_mask_path']))>0
        assert not (context_mask & object_mask).any() and not (context_mask & ~valid).any()
        assert contextual['objects'][0]['reference_frame']=='frame_0001'
        assert contextual['objects'][1]['selectable'] is False
        assert len(contextual['observed_regions'])==3 and len(contextual['unavailable_regions'])==1
        region_faces=[np.frombuffer(gzip.decompress((root/'context-result'/r['faces']['asset']['path']).read_bytes()),dtype='<u4')
                      for r in contextual['observed_regions'] if r['faces']]
        assert len(np.intersect1d(*region_faces))>0
        for region in contextual['observed_regions']:
            mask_value=np.asarray(Image.open(root/'context-result'/region['mask']['path']))>0
            observed=points[mask_value & valid]
            assert np.allclose(region['bounds_native']['min'],observed.min(0))
            assert np.allclose(region['bounds_native']['max'],observed.max(0))
            assert np.isclose(region['ground_extent_native']['min'],(observed@[0,-1,0]).min())
            if region['id']=='covered-region':
                assert region['faces'] is None and region['surface_status']=='covered_by_generated'
                continue
            face_ids=np.frombuffer(gzip.decompress((root/'context-result'/region['faces']['asset']['path']).read_bytes()),dtype='<u4')
            assert len(np.unique(face_ids))==len(face_ids) and face_ids.max()<contextual['objects'][-1]['mesh']['index_count']//3
        context_revision(root,root/'context-result',root/'context-revision',ehs_repo)
        revised=json.loads((root/'context-revision/scene.json').read_text())
        assert len(revised['objects'])==3 and revised['observed_regions']==contextual['observed_regions']
        assert (root/'context-revision/scene.bin').read_bytes()==(root/'context-result/scene.bin').read_bytes()
        assert before=={p.relative_to(root/'result').as_posix():digest(p) for p in (root/'result').rglob('*') if p.is_file()}
        frame['input_to_canonical_pixel_centres']=[[1,0,0],[0,1,0],[0,0,1]]
        write_json(root/'manifest.json',manifest)
        try:
            context_revision(root,root/'result',root/'wrong-affine-context',ehs_repo)
        except ValueError as exc:
            assert 'affine differs' in str(exc)
        else:
            raise AssertionError('Changed source-photo affine must not silently reproject the object')
    print('PASS: pose/assembly/GLB; source binding, full-photo affine/EXIF, shared overlapping regions/bounds, context revision and frozen artifacts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--observed-self-check', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--source-run', type=Path, help='Normal product run containing the complete object evidence registry')
    parser.add_argument('--observed-output', type=Path, help='New native observed-only result directory; no object generation')
    parser.add_argument('--ehs-repo', type=Path, help='Existing product helpers for exact source-mask decoding and pixel mapping')
    parser.add_argument('--add-context', action='store_true', help='Add context to a newly assembled job before packing')
    parser.add_argument('--source-result', type=Path, help='Frozen result to copy for a context-only revision')
    parser.add_argument('--context-output', type=Path, help='New result directory; preserves source comparison and mesh bytes')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--eval-size', type=int, default=288)
    args = parser.parse_args()
    if (args.self_check or args.observed_self_check or args.add_context or args.context_output or args.source_run) and not args.ehs_repo:
        parser.error('--ehs-repo is required for the existing source-mask helpers')
    if bool(args.source_run)!=bool(args.observed_output):
        parser.error('--source-run and --observed-output are required together')
    if args.self_check:
        self_check(args.ehs_repo)
    elif args.observed_self_check:
        observed_self_check(args.ehs_repo)
    elif args.source_run:
        print(json.dumps(assemble_observed_scene(args.source_run,args.observed_output,args.ehs_repo),indent=2))
    elif args.run_dir and args.context_output:
        print(json.dumps(context_revision(args.run_dir,args.source_result or args.run_dir/'result',args.context_output,args.ehs_repo),indent=2))
    elif args.run_dir and args.add_context:
        print(json.dumps(add_scene_context(args.run_dir,args.run_dir/'result',args.ehs_repo),indent=2))
    elif args.run_dir:
        assemble(args.run_dir, args.iterations, args.eval_size)
    else:
        parser.error('Supply --run-dir or --self-check')
