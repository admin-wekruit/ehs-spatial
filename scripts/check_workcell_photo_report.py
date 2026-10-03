"""Check actual shared-report schema, coverage, coordinate transform and model references."""
import json
import sys
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.platform.contracts import validate_document, Revision
from scripts.workcell_photo_oneshot import _array, _frame

root = Path(sys.argv[1])
report = json.loads((root/'scene-report.json').read_text())
doc = report['revision']['document']; validate_document(doc); Revision.model_validate(report['revision'])
objects = {o['id']:o for o in report['objects']}
# The capture point cloud is source context, not a catalog object.
entities = [e for e in doc['entities'] if not e.get('sourceContext')]
assert {e['id'] for e in entities} == set(objects)
assert 'emergency-button' in objects
if '--require-lights' in sys.argv:
    assert sum(o['kind']=='signal light' for o in objects.values()) >= 2, 'Known capture requires both entrance lamps'
    assert not report['coverage']['categoriesNotSegmented']
T=np.asarray(report['sceneTransformNative'])
assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6)
assert np.isclose(np.linalg.det(T[:3,:3]),1)
for i,camera in enumerate(doc['cameras'],1):
    assert doc['geometryBindings'][camera['imageId']]['cameraId'] == camera['id']
    f=_frame(root,i);valid=_array(f['non_ambiguous_mask']).astype(bool)
    points=_array(f['pts3d'])[valid][::500];old_pose=_array(f['camera_poses']);new_pose=np.array(camera['cameraToWorld'])
    before=(points-old_pose[:3,3])@old_pose[:3,:3]
    after=(trimesh.transform_points(points,T)-new_pose[:3,3])@new_pose[:3,:3]
    assert np.allclose(before,after,atol=3e-5)
    assert np.allclose(camera['K'],_array(f['intrinsics']))
for entity in entities:
    item=objects[entity['id']]
    distance=item.get('groundDistance',{})
    if len({o['photo'] for o in item['observations']})<2:
        assert not distance.get('byPhoto'), 'single-view distance remains unsupported'
    for sample in [*distance.get('byPhoto',{}).values(), *([distance['feature']] if distance.get('feature') else [])]:
        if sample['valueNative'] is None: continue
        point,foot=np.asarray(sample['pointNative']),np.asarray(sample['footNative'])
        assert abs(foot[2])<1e-6 and np.allclose(point[:2],foot[:2],atol=1e-6)
        assert np.isclose(np.linalg.norm(point-foot),sample['valueNative'],atol=1e-6)
    assert entity['physicalDimensionsUnknown'] and entity['modelOrientationUnknown']
    for evidence in entity['measurements'].get('orientationEvidence', {}).values():
        if evidence.get('axisNative') is not None:
            axis=np.asarray(evidence['axisNative'])
            tilt=np.degrees(np.arccos(np.clip(abs(axis[2])/np.linalg.norm(axis),0,1)))
            assert np.isclose(tilt,evidence['valueDeg'],atol=1e-5)
    if len({o['photo'] for o in item['observations']})<=1:
        assert entity['physicalDimensionsUnknown']
        assert not entity['measurements']
        assert item['visibleHeightNative'] is None
    for rep in [*entity['representations'],*entity['modelVariants'].values()]:
        if rep['kind'] != 'generated_mesh':
            continue  # selectable source-point subsets carry frame references, not model nodes
        path=root/report['assetURLs'][rep['assetId']]
        exported=trimesh.load(path,force='scene')
        mesh=exported.to_geometry()
        assert np.isfinite(mesh.vertices).all() and len(mesh.faces)>0
        assert np.allclose(mesh.bounds,[rep['bounds']['min'],rep['bounds']['max']],atol=1e-5)
        source=rep['sourceRefs'][0]; scene=trimesh.load(root/source['file'],force='scene')
        parts=[scene.geometry[scene.graph[node][1]] for node in source['nodes']]
        if all(p.visual.kind=='texture' and 'color' in p.visual.vertex_attributes for p in parts):
            output=next(iter(exported.geometry.values()))
            assert np.array_equal(trimesh.visual.color.to_rgba(output.visual.vertex_attributes['color']),np.concatenate([trimesh.visual.color.to_rgba(p.visual.vertex_attributes['color']) for p in parts])), 'photo vertex colors must survive report export'
            if all(p.visual.material.doubleSided for p in parts):
                assert output.visual.material.doubleSided, 'two-sided sheets must survive report export'
    if entity['id']=='floor':
        assert max(abs(mesh.vertices[:,2]+entity['currentModelTransform']['position'][2]))<1e-5
    for obs in item['observations']:
        c=doc['cameras'][obs['photo']-1];x0,y0,x1,y1=obs['box']
        assert 0<=x0<x1<=c['width'] and 0<=y0<y1<=c['height']
# Visible extents: this revision's frames and floor, each observation's objects-stage support (never its outline).
import shutil, tempfile
from scripts.workcell_photo_objects import BUILD_OUTPUTS, build as build_catalog, fence_panel_supported, review_fence_observations, support_inputs_missing
missing=support_inputs_missing(root,list(objects.values()))
if missing:
    # Without the objects-stage inputs the report must say its extents are objects-stage values, and which inputs are absent.
    for item in objects.values():
        extent=item['visibleExtentFloor']
        assert extent['status']=='objects_stage' and extent['missingSupportInputs']==missing, (item['id'],extent)
    print('NOTE: support inputs missing',missing,'- visible extents are labelled objects-stage values; re-measure not checked')
else:
    # Independent of the report's own re-measure: rebuild the catalog with the objects stage itself on this revision's
    # frames and floor, then every report observation must equal the rebuilt one exactly.
    with tempfile.TemporaryDirectory() as directory:
        rebuilt_root=Path(directory)
        for path in root.iterdir():
            if path.is_file() and path.name not in BUILD_OUTPUTS:
                (rebuilt_root/path.name).symlink_to(path)
        rebuilt={item['id']:item for item in build_catalog(rebuilt_root,[root/f'photo-{i}.png' for i in range(1,5)],proxy_textures=False)['objects']}
    for item in objects.values():
        assert item['visibleExtentFloor']['status']=='remeasured' and item['visibleExtentFloor']['floor']=="this revision's floor", item['id']
        if item['visibleExtentFloor']['angleToRevisionFloorDeg'] is not None:
            assert abs(item['visibleExtentFloor']['angleToRevisionFloorDeg'])<1e-6
        twin=rebuilt[item['id']]['observations']
        assert [(o['photo'],o['source']) for o in item['observations']]==[(o['photo'],o['source']) for o in twin], ('observation identity',item['id'])
        for got,want in zip(item['observations'],twin):
            assert (got['supportedPixels'],got['maskPixels'])==(want['supportedPixels'],want['maskPixels']), (item['id'],got['photo'])
            assert json.dumps(got['observedMeasurements'],sort_keys=True)==json.dumps(want['observedMeasurements'],sort_keys=True), (item['id'],got['photo'])
            if item['kind']=='safety fence':
                assert want['supportedPixels']>0
    # The catalog review is settled: rerunning it on a copy removes nothing and rewrites nothing.
    with tempfile.TemporaryDirectory() as directory:
        copy=Path(directory)
        for name in ['objects.json','geometry.json','sam3.json',*(p.name for p in root.glob('frame_*.json.gz'))]:
            shutil.copyfile(root/name,copy/name)
        before=(copy/'objects.json').read_bytes()
        assert review_fence_observations(copy)==[] and (copy/'objects.json').read_bytes()==before, 'catalog still holds an unsupported fence observation'
structural=report.get('experiment',{}).get('structuralModel')
if structural and structural['promotionAllowed']:
    from ehs_spatial.platform.scene_measurements import fitted_bend
    assert structural['measurementAngleDeg'] is None
    scene=trimesh.load(root/'workcell-metric.glb',force='scene')
    for side in ('left','right'):
        entity_id='v-guard-'+side
        annotation=next(row for row in report['bendAnalysis']['items'] if row['entityId']==entity_id)
        assert abs(annotation['result']['value']-structural['sharedAngleDeg'])<.002
        source=trimesh.load(root/objects[entity_id]['model']['file'],force='scene')
        triangles=[]
        for node in scene.graph.nodes_geometry:
            if not node.startswith(entity_id+':'): continue
            matrix,geometry=scene.graph[node]; original=source.geometry[source.graph[node.split(':',1)[1]][1]]
            assert np.array_equal(scene.geometry[geometry].visual.vertex_attributes['color'],original.visual.vertex_attributes['color']), 'metric download must retain source colors'
            part=scene.geometry[geometry].copy(); part.apply_transform(matrix)
            triangles.append(part.triangles)
        assert abs(fitted_bend(np.concatenate(triangles))['value']-structural['sharedAngleDeg'])<.002, 'download and displayed angle must use the same geometry'
print('PASS: shared schema, object coverage, source-camera transform, model bounds, unknown dimensions, visible extents equal to an objects-stage rebuild on this revision (or labelled objects-stage when inputs are absent), catalog review settled;',len(objects),'objects')
