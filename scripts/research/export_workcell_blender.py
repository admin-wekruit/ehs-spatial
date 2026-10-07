"""Export the frozen workcell into native Blender, without model inference.

Run with Blender --background --python this_file -- --root EXPERIMENT.
Add --verify to reopen the saved file and check geometry and camera projections.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def floor_transform(floor):
    plane = np.asarray(floor['plane_native'], dtype=float)
    plane /= np.linalg.norm(plane[:3])
    # Same convention as ehs_spatial.geometry: origin=-d*n; rotate n to +Z.
    origin = -plane[3] * plane[:3]
    rotation = Vector(plane[:3]).rotation_difference(Vector((0, 0, 1))).to_matrix()
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation)
    transform[:3, 3] = -transform[:3, :3] @ origin
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
    return transform


def buffers(binary, entry):
    spec = entry['mesh']
    assert spec['stride'] == 9 and spec['index_type'] == 'uint32'
    vertices = np.frombuffer(binary, '<f4', spec['vertex_count'] * 9, spec['byte_offset']).reshape(-1, 9)
    indices = np.frombuffer(binary, '<u4', spec['index_count'], spec['index_byte_offset'])
    assert np.isfinite(vertices).all() and len(indices) % 3 == 0
    assert len(indices) and indices.max() < len(vertices)
    return vertices, indices


def native_pose(root, entry):
    if entry['source'] == 'observed':
        return np.eye(4)
    return np.asarray(read(root / 'result' / entry['metrics']['comparison'])['final_object_to_world'])


def linear_color(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def make_mesh(entry, vertices, indices, material, collection):
    mesh = bpy.data.meshes.new(entry['id'])
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set('co', vertices[:, :3].ravel())
    mesh.loops.add(len(indices))
    mesh.loops.foreach_set('vertex_index', indices)
    mesh.polygons.add(len(indices) // 3)
    mesh.polygons.foreach_set('loop_start', np.arange(0, len(indices), 3, dtype=np.int32))
    mesh.polygons.foreach_set('loop_total', np.full(len(indices) // 3, 3, dtype=np.int32))
    mesh.polygons.foreach_set('use_smooth', np.ones(len(indices) // 3, dtype=bool))
    mesh.update()
    rgba = np.column_stack((linear_color(vertices[:, 6:9]), np.ones(len(vertices)))).astype(np.float32)
    colors = mesh.color_attributes.new(name='Observed or generated color', type='FLOAT_COLOR', domain='POINT')
    colors.data.foreach_set('color', rgba.ravel())
    mesh.materials.append(material)
    obj = bpy.data.objects.new(entry['id'], mesh)
    collection.objects.link(obj)
    for key in ('id', 'label', 'source', 'model'):
        obj['object_id' if key == 'id' else key] = entry[key]
    obj['source_inventory_indices_json'] = json.dumps(entry.get('source_inventory_indices', []))
    obj['metric_scale_known'] = False
    return obj


def activate_camera(scene, camera):
    scene.camera = camera
    scene.render.resolution_x = camera['image_width']
    scene.render.resolution_y = camera['image_height']
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = camera['pixel_aspect_y']


def make_camera(spec, transform, collection):
    K = np.asarray(spec['K'])
    width, height = spec['width'], spec['height']
    data = bpy.data.cameras.new(spec['id'])
    data.sensor_fit = 'HORIZONTAL'
    # ponytail: sensor width is a projection convention, not measured hardware.
    data.sensor_width = 36
    data.lens = K[0, 0] * data.sensor_width / width
    aspect = float(K[0, 0] / K[1, 1])
    data.shift_x = (width / 2 - (K[0, 2] + .5)) / width
    data.shift_y = ((K[1, 2] + .5) - height / 2) * aspect / width
    data.clip_start, data.clip_end = .001, 100
    obj = bpy.data.objects.new(spec['id'], data)
    collection.objects.link(obj)
    obj.matrix_world = Matrix(transform @ np.asarray(spec['camera_to_world']) @ np.diag([1, -1, -1, 1]))
    obj['source_K_json'] = json.dumps(spec['K'])
    obj['source_camera_to_world_json'] = json.dumps(spec['camera_to_world'])
    obj['image_width'], obj['image_height'], obj['pixel_aspect_y'] = width, height, aspect
    obj['parameter_source'] = 'Pi3X estimated pinhole camera; not physical camera calibration'
    obj['image_reference'] = spec['image']
    return obj


def add_post_trial(root, scene, assets, transform):
    parameters = read(root / 'blender-01/post-fit/parameters.json')
    comparisons = read(root / 'blender-01/post-fit/comparisons.json')
    trial = scene.copy()
    trial.name = '02 · 参数化防撞柱试验'
    trial.use_fake_user = True  # Keep the inactive comparison scene when saving.
    trial.collection.children.unlink(assets)
    collection = bpy.data.collections.new('TRIAL · cylinders fitted from photographs')
    trial.collection.children.link(collection)
    for obj in assets.objects:
        if obj.get('object_id') not in ('left_post', 'right_post'):
            collection.objects.link(obj)
    bpy.context.window.scene = trial
    for spec in parameters['objects']:
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1, depth=1, end_fill_type='TRIFAN')
        obj = bpy.context.object
        for parent in list(obj.users_collection):
            parent.objects.unlink(obj)
        collection.objects.link(obj)
        obj.name = spec['object_id'] + ' · parametric'
        obj['object_id'] = spec['object_id']
        obj['source'] = 'fitted cylinder prior; no footplate'
        obj['metric_scale_known'] = False
        obj['radius_native'] = spec['fitted_parameters']['radius_native']
        obj['height_native'] = spec['fitted_parameters']['height_native']
        obj.matrix_world = Matrix(transform @ np.asarray(spec['native_object_to_world']))
        for axis, prop in enumerate(('radius_native', 'radius_native', 'height_native')):
            driver = obj.driver_add('scale', axis).driver
            variable = driver.variables.new()
            variable.name = 'parameter'
            variable.targets[0].id = obj
            variable.targets[0].data_path = f'["{prop}"]'
            driver.expression = 'parameter'
            obj.id_properties_ui(prop).update(min=1e-6, description='Fitted Pi3X relative units, not metres')
        colors = obj.data.color_attributes.new(name='Cylinder display color', type='FLOAT_COLOR', domain='POINT')
        colors.data.foreach_set('color', np.tile([.012, .012, .012, 1], len(obj.data.vertices)))
        obj['color_source'] = 'Uniform dark display color; not inferred physical material'
        obj.data.polygons.foreach_set('use_smooth', [len(poly.vertices) == 4 for poly in obj.data.polygons])
    trial['trial_parameters_json'] = json.dumps(parameters, ensure_ascii=False)
    trial['trial_comparisons_json'] = json.dumps(comparisons, ensure_ascii=False)
    bpy.data.texts.new('POST_FIT_PARAMETERS.json').write(json.dumps(parameters, ensure_ascii=False, indent=2))
    bpy.data.texts.new('POST_FIT_COMPARISONS.json').write(json.dumps(comparisons, ensure_ascii=False, indent=2))
    bpy.context.window.scene = scene


def verify_post_trial(root, transform):
    baseline = bpy.context.scene
    trial = bpy.data.scenes['02 · 参数化防撞柱试验']
    bpy.context.window.scene = trial
    parameters = read(root / 'blender-01/post-fit/parameters.json')
    records = []
    for spec in parameters['objects']:
        obj = next(o for o in trial.objects if o.get('object_id') == spec['object_id'])
        bpy.context.view_layer.update()
        expected = transform @ np.asarray(spec['native_object_to_world'])
        # Read the evaluated object: another scene's original matrix cache can
        # remain identity after file load even though its drivers evaluate.
        def evaluated():
            return obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        assert np.allclose(np.asarray(evaluated().matrix_world), expected, atol=1e-6)
        coords = np.array([v.co[:] for v in obj.data.vertices])
        radius = np.linalg.norm(coords[:, :2], axis=1)
        ring = coords[radius > .5]
        assert np.allclose(np.linalg.norm(ring[:, :2], axis=1), 1, atol=1e-6)
        assert np.allclose(np.abs(coords[:, 2]), .5, atol=1e-6)
        angles = np.arctan2(ring[:, 1], ring[:, 0]) * 64 / (2 * np.pi)
        assert np.max(np.abs(angles - np.round(angles))) < 1e-4
        original = obj['radius_native']
        obj['radius_native'] = original * 1.1
        obj.update_tag()
        bpy.context.view_layer.update()
        assert np.allclose(evaluated().scale[:2], [original * 1.1] * 2, atol=1e-6)
        obj['radius_native'] = original
        obj.update_tag()
        bpy.context.view_layer.update()
        height = obj['height_native']
        obj['height_native'] = height * 1.1
        obj.update_tag()
        bpy.context.view_layer.update()
        assert abs(evaluated().scale.z - height * 1.1) < 1e-6
        obj['height_native'] = height
        obj.update_tag()
        bpy.context.view_layer.update()
        obj.data.calc_loop_triangles()
        records.append({'object_id': spec['object_id'], 'radius_driver_edit_check': 'passed',
                        'height_driver_edit_check': 'passed',
                        'triangles': len(obj.data.loop_triangles)})
    bpy.context.window.scene = baseline
    return records


def verify(root, scene_spec, binary, transform):
    scene = bpy.context.scene
    records = []
    for entry in scene_spec['objects']:
        obj = next(o for o in scene.objects if o.get('object_id') == entry['id'])
        vertices, indices = buffers(binary, entry)
        assert len(obj.data.vertices) == len(vertices)
        assert len(obj.data.polygons) == len(indices) // 3
        coords = np.empty((len(vertices), 3), dtype=np.float32)
        obj.data.vertices.foreach_get('co', coords.ravel())
        assert np.array_equal(coords, vertices[:, :3]), entry['id']
        actual_indices = np.empty(len(indices), dtype=np.uint32)
        obj.data.loops.foreach_get('vertex_index', actual_indices)
        assert np.array_equal(actual_indices, indices), entry['id']
        pose = transform @ native_pose(root, entry)
        expected = vertices[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        actual_pose = np.asarray(obj.matrix_world, dtype=float)
        actual = coords @ actual_pose[:3, :3].T + actual_pose[:3, 3]
        error = float(np.max(np.abs(actual - expected)))
        assert error < 5e-6, (entry['id'], error)
        records.append({'object_id': entry['id'], 'triangles': len(indices) // 3, 'max_world_coordinate_error_native': error})
    assert len([o for o in scene.objects if o.get('object_id')]) == len(scene_spec['objects'])
    camera_records = []
    for spec in scene_spec['cameras']:
        cam = bpy.data.objects[spec['id']]
        activate_camera(scene, cam)
        bpy.context.view_layer.update()
        K, c2w = np.asarray(spec['K']), np.asarray(spec['camera_to_world'])
        residuals = []
        for point in [(-.2, -.25, 1), (.25, .15, 1), (0, 0, 2), (.4, -.3, 3)]:
            p = np.asarray(point)
            native = c2w[:3, :3] @ p + c2w[:3, 3]
            world = transform[:3, :3] @ native + transform[:3, 3]
            ndc = world_to_camera_view(scene, cam, Vector(world))
            projected = np.array([ndc.x * spec['width'] - .5, (1 - ndc.y) * spec['height'] - .5])
            expected = (K @ p)[:2] / p[2]
            residuals.append(float(np.max(np.abs(projected - expected))))
        assert max(residuals) < .001, (spec['id'], residuals)
        camera_records.append({'camera': spec['id'], 'max_conversion_error_pixels': max(residuals)})
    points = np.load(root / 'evidence/floor_points.npy', allow_pickle=False)
    signed_heights = points @ transform[2, :3] + transform[2, 3]
    floor = read(root / 'evidence/floor.json')
    p95 = float(np.quantile(np.abs(signed_heights), .95))
    assert abs(p95 - floor['all_point_residual_p95_native']) < 1e-5
    assert scene.unit_settings.system == 'NONE' and not scene['metric_scale_known']
    return {'status': 'passed', 'objects': records, 'cameras': camera_records,
            'parametric_trial': verify_post_trial(root, transform),
            'source_triangle_count': sum(r['triangles'] for r in records),
            'floor_residual_p95_native': p95, 'vlm_calls_this_export': 0, 'new_model_inference_calls': 0}


def build(root, output, scene_spec, binary, transform):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = '01 · 原始重建资产'
    scene.unit_settings.system = 'NONE'
    scene['run_id'] = scene_spec['run_id']
    scene['metric_scale_known'] = False
    scene['units_explanation'] = 'Pi3X native relative units; no metre calibration'
    scene['native_to_blender_json'] = json.dumps(transform.tolist())
    scene['blender_to_native_json'] = json.dumps(np.linalg.inv(transform).tolist())
    assets = bpy.data.collections.new('WORKCELL · editable objects')
    cameras = bpy.data.collections.new('CAMERAS · three source views')
    scene.collection.children.link(assets)
    scene.collection.children.link(cameras)
    mat = bpy.data.materials.new('Source vertex colors')
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    color = mat.node_tree.nodes.new('ShaderNodeVertexColor')
    color.layer_name = 'Observed or generated color'
    mat.node_tree.links.new(color.outputs['Color'], shader.inputs['Base Color'])
    shader.inputs['Roughness'].default_value = .75
    for entry in scene_spec['objects']:
        vertices, indices = buffers(binary, entry)
        obj = make_mesh(entry, vertices, indices, mat, assets)
        obj.matrix_world = Matrix(transform @ native_pose(root, entry))
        obj['native_object_to_world_json'] = json.dumps(native_pose(root, entry).tolist())
        if entry['source'] == 'generated':
            obj['comparison_record'] = entry['metrics']['comparison']
        print('IMPORTED', entry['id'], len(indices) // 3, flush=True)
    for spec in scene_spec['cameras']:
        make_camera(spec, transform, cameras)
    scene.world = bpy.data.worlds.new('Studio background · display only')
    scene.world.color = (.2, .2, .2)
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'VERTEX'
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = 'BOTH'
    scene.display.shading.background_type = 'WORLD'
    scene.display.shading.show_specular_highlight = True
    scene.view_settings.view_transform = 'Standard'
    scene.render.image_settings.file_format = 'PNG'
    activate_camera(scene, bpy.data.objects['frame_0003'])
    add_post_trial(root, scene, assets, transform)
    provenance = {
        'run_id': scene_spec['run_id'], 'blender_version': bpy.app.version_string,
        'source_files_sha256': {p: sha(root / p) for p in ('result/scene.json', 'result/scene.bin', 'evidence/floor.json', 'evidence/objects.json', 'manifest.json')},
        'native_to_blender': transform.tolist(), 'blender_to_native': np.linalg.inv(transform).tolist(),
        'coordinate_convention': 'Plane point nearest the native origin becomes origin; plane normal rotated to +Z. Rigid conversion only, no scaling.',
        'floor': read(root / 'evidence/floor.json'),
        'parameters': {
            'floor_plane': 'Pi3X points selected by SAM floor masks and manual interior ROIs; RANSAC plus SVD',
            'floor_extent': 'Observed frame_0003 floor mask triangulation; no inferred factory boundary or thickness',
            'camera_K': 'Pi3X pinhole fit from local point directions; not measured intrinsics',
            'camera_pose': 'Joint three-view Pi3X geometry',
            'object_geometry': 'RecGen generated mesh; hidden surfaces are inferred',
            'object_pose': 'RecGen anchor pose composed with Pi3X camera; refined numerically against multi-view masks and depth',
            'physical_scale': 'UNKNOWN; scene units intentionally NONE',
            'mass_friction_floor_thickness_robot_joints': 'UNKNOWN; not assigned',
            'studio_lighting': 'Display-only setting; not inferred factory lighting'},
        'post_fit_parameters_sha256': sha(root / 'blender-01/post-fit/parameters.json'),
        'post_fit_comparisons_sha256': sha(root / 'blender-01/post-fit/comparisons.json'),
        'vlm_calls_this_export': 0, 'new_model_inference_calls': 0,
        'source_objects': scene_spec['objects'], 'source_cameras': scene_spec['cameras']}
    write(output / 'parameters.json', provenance)
    bpy.data.texts.new('PARAMETERS.json').write(json.dumps(provenance, ensure_ascii=False, indent=2))
    instructions = '''工位 Blender 场景 / lucida-replica-01

10 个独立网格：9 个生成对象 + 观测地面。保留全部源三角面，可独立编辑。
CAMERAS 中是三张照片估计的相机；精确切换请选中相机后运行 activate_source_view.py。
地面已转为 Z 向上，Z=0 是拟合参考平面，观测地面的微小起伏仍保留。
单位为模型相对单位，不是米。尺寸、质量、摩擦和机器人关节没有实测数据。
PARAMETERS.json 保存参数来源、原始矩阵、逆变换和源文件 SHA256。
原始照片未打包。Blender 中的 studio 光照只是为了查看几何。
这次导出没有新增 VLM/3D 模型推理；它没有改善或重新生成原有资产形状。
顶部 Scene 下拉菜单可切换“02 · 参数化防撞柱试验”。它用拟合圆柱替换两柱作对照。
选中 parametric 柱，在 Object Properties → Custom Properties 调 radius_native/height_native。
这两个属性驱动真实柱体半径和高度；位置通过常规 Location 修改。
该试验不表示底板等细节；右柱深度误差增加，原始场景仍保留完整资产。
'''
    bpy.data.texts.new('README · 参数从哪里来').write(instructions)
    # Per-camera pixel aspect is a scene setting in Blender, so switching source
    # cameras also applies their exact K-derived aspect. No persistent addon.
    switch = '''import bpy
scene = bpy.context.scene
cam = bpy.context.active_object
assert cam and cam.type == 'CAMERA' and 'pixel_aspect_y' in cam, 'Select a source camera first'
scene.camera = cam
scene.render.resolution_x = cam['image_width']
scene.render.resolution_y = cam['image_height']
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = 1
scene.render.pixel_aspect_y = cam['pixel_aspect_y']
'''
    bpy.data.texts.new('activate_source_view.py').write(switch)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
                area.spaces.active.shading.color_type = 'VERTEX'
                area.spaces.active.overlay.show_extras = False
                area.spaces.active.clip_start = .001
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['robot'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['robot']
    bpy.ops.wm.save_as_mainfile(filepath=str(output / 'workcell.blend'), compress=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    started = time.perf_counter()
    root = args.root.resolve()
    output = root / 'blender-01'
    output.mkdir(exist_ok=True)
    scene_spec = read(root / 'result/scene.json')
    binary = (root / 'result/scene.bin').read_bytes()
    transform = floor_transform(read(root / 'evidence/floor.json'))
    if args.verify:
        bpy.ops.wm.open_mainfile(filepath=str(output / 'workcell.blend'))
    else:
        build(root, output, scene_spec, binary, transform)
    checks = verify(root, scene_spec, binary, transform)
    if args.render:
        for spec in scene_spec['cameras']:
            activate_camera(bpy.context.scene, bpy.data.objects[spec['id']])
            bpy.context.scene.render.filepath = str(output / f"{spec['id']}.png")
            bpy.ops.render.render(write_still=True)
        baseline = bpy.context.scene
        trial = bpy.data.scenes['02 · 参数化防撞柱试验']
        bpy.context.window.scene = trial
        activate_camera(trial, bpy.data.objects['frame_0003'])
        trial.render.filepath = str(output / 'parametric-frame_0003.png')
        bpy.ops.render.render(write_still=True)
        bpy.context.window.scene = baseline
    checks['reopened_from_disk'] = args.verify
    checks['elapsed_seconds'] = time.perf_counter() - started
    checks['blend_sha256'] = sha(output / 'workcell.blend')
    write(output / ('verification.json' if args.verify else 'build-check.json'), checks)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
