"""Run: python scripts/check_workcell_photo_objects.py OUTPUT_DIRECTORY."""
import json
from pathlib import Path
import sys
import tempfile

from PIL import Image

import cv2
import numpy as np
import trimesh

from workcell_photo_objects import _inside, _same, _project, _observation, build
from workcell_photo_oneshot import _array, _frame


def check(root):
    root=Path(root)
    data=json.loads((root/'objects.json').read_text())
    assert data['schemaVersion']==1
    objects=data['objects']
    assert len({o['id'] for o in objects})==len(objects)
    scenes={}
    frames={i:_frame(root,i) for i in (1,2,3,4)}
    for raw in frames.values():
        points=_array(raw['pts3d']); valid=_array(raw['non_ambiguous_mask']).astype(bool)&np.isfinite(points).all(2)
        yy,xx=np.where(valid)
        uv,depth=_project(points[valid],{'pose':_array(raw['camera_poses']),'K':_array(raw['intrinsics'])})
        assert np.all(depth>0) and np.median(np.linalg.norm(uv-np.column_stack((xx,yy)),axis=1))<3
    linked=set()
    for obj in objects:
        assert obj['observations'], f"No photo evidence: {obj['id']}"
        for model in [obj['model'], *obj.get('modelsByPhoto',{}).values()]:
            if model is None:
                assert obj['representation'].startswith('unknown geometry')
                assert all(m['valueNative'] is None for m in obj['measurements'].values())
                continue
            assert Path(model['file']).name==model['file'], 'Model must be local to report'
            if model['file'] not in scenes:
                scenes[model['file']]=trimesh.load(root/model['file'],force='scene')
            scene=scenes[model['file']]
            assert model['nodes'] and len(set(model['nodes']))==len(model['nodes'])
            for node in model['nodes']:
                assert node in scene.graph.nodes_geometry, (obj['id'],node)
                mesh=scene.geometry[scene.graph[node][1]]
                assert len(mesh.faces)>0 and np.isfinite(mesh.vertices).all()
                linked.add((model['file'],node))
        photos=set()
        for observation in obj['observations']:
            photo=observation['photo']; photos.add(photo)
            assert photo in (1,2,3,4)
            h,w=frames[photo]['image']['shape'][:2]
            x0,y0,x1,y1=observation['box']
            assert 0<=x0<x1<=w and 0<=y0<y1<=h,(obj['id'],observation)
            assert observation['polygons']
            for polygon in observation['polygons']:
                polygon=np.asarray(polygon)
                assert len(polygon)>=3 and polygon.shape[1]==2 and np.isfinite(polygon).all()
                assert ((polygon>=0)&(polygon<=[w,h])).all()
        assert set(obj['measurements'])=={'height','width','depth','groundClearance'}
        for measurement in obj['measurements'].values():
            assert measurement['source'] and measurement['status']
            value=measurement['valueNative']
            assert value is None or np.isfinite(value)
            if len(photos)==1:
                assert value is None, 'Single-view model dimensions must not masquerade as physical measurements'
            if value is not None:
                assert measurement['status'] in ('input-hypothesis','conditional-model-estimate')
        if obj['representation'].startswith('observed surface'):
            assert all(m['valueNative'] is None for m in obj['measurements'].values())
    extras=scenes['object-extras.glb']
    assert all(('object-extras.glb',n) in linked for n in extras.graph.nodes_geometry)
    assert all(v<3 for v in data['coverage']['projectionMedianPixels'].values())
    # All disconnected fragments must survive independently, without filling
    # the gap between them; tiny fragments retain their own pixel footprints.
    mask=np.zeros((12,16),bool); mask[1:5,1:5]=True; mask[6:10,10:15]=True
    observed=_observation(mask,1,'disconnected-fragment regression')
    assert len(observed['polygons'])==2
    reconstructed=np.zeros(mask.shape,np.uint8)
    cv2.fillPoly(reconstructed,[np.asarray(p,np.int32) for p in observed['polygons']],1)
    assert np.array_equal(reconstructed.astype(bool),mask)
    tiny=np.zeros((6,6),bool); tiny[1,1]=True; tiny[4,4]=True
    fragments=_observation(tiny,1,'single-pixel regression')['polygons']
    assert len(fragments)==2 and all(cv2.contourArea(np.asarray(p,np.int32))==1 for p in fragments)
    # Same category in one photo is insufficient evidence to merge distinct instances.
    frame={'pose':np.eye(4),'K':np.eye(3)}
    a={'photo':1,'mask':np.pad(np.ones((2,2),bool),((0,2),(0,2))), 'points':np.array([[0,0,1],[1,1,1]])}
    b={'photo':1,'mask':np.pad(np.ones((2,2),bool),((2,0),(2,0))), 'points':np.array([[2,2,1],[3,3,1]])}
    assert not _same(a,b,{1:frame})
    assert _same(a,a,{1:frame})
    b['photo']=2
    assert not _same(a,b,{1:frame,2:frame})
    assert _inside(np.array([[0,0,1],[1,1,1]]),a,{1:frame})==1
    assert _same(a,dict(a,photo=2),{1:frame,2:frame})
    try:
        build(root,[])
    except ValueError as error:
        assert 'Four distinct source photos' in str(error)
    else:
        raise AssertionError('Invalid source inputs accepted')
    # Exercise the complete build with one depth-supported pixel and one pixel
    # without valid depth: both detections need cards, neither permits a mesh.
    with tempfile.TemporaryDirectory(prefix='photo-object-retention-') as temporary:
        output=Path(temporary)
        for path in root.iterdir():
            if path.is_file() and path.name not in ('sam3.json','objects.json','object-extras.glb'):
                (output/path.name).symlink_to(path.resolve())
        segmentation=json.loads((root/'sam3.json').read_text())
        words=[p['text'] for p in segmentation['prompts']]
        if 'instruction poster' not in words:
            segmentation['prompts'].append({'text':'instruction poster'})
            for row in segmentation['results']: row.append({'rle':[],'scores':[]})
        j=[p['text'] for p in segmentation['prompts']].index('instruction poster')
        for row in segmentation['results']: row[j]={'rle':[],'scores':[]}
        valid=_array(frames[1]['non_ambiguous_mask']).astype(bool)
        assert valid.any() and (~valid).any(), 'Retention check needs valid and invalid depth samples'
        for y,x in [np.argwhere(valid)[0],np.argwhere(~valid)[0]]:
            offset=int(x*valid.shape[0]+y)
            segmentation['results'][0][j]['rle'].append(json.dumps({'size':list(valid.shape),'counts':[offset,1,int(valid.size-offset-1)]}))
            segmentation['results'][0][j]['scores'].append(.9)
        (output/'sam3.json').write_text(json.dumps(segmentation))
        photos=[]
        for i,raw in frames.items():
            path=output/f'test-source-{i}.png'; Image.fromarray(_array(raw['image'])).save(path); photos.append(path)
        rebuilt=build(output,photos)
        posters=[o for o in rebuilt['objects'] if o['label'].startswith('Instruction Poster')]
        assert len(posters)==2 and all(o['model'] is None and len(o['observations'])==1 for o in posters)
        assert sorted(o['observations'][0]['supportedPixels'] for o in posters)==[0,1]
        assert all(all(m['valueNative'] is None for m in o['measurements'].values()) for o in posters)
        rejected=rebuilt['coverage']['unmeshedDetections']
        assert all(any(r['objectId']==o['id'] for r in rejected) for o in posters)
    print(json.dumps({'status':'passed','objects':len(objects),'linkedMeshNodes':len(linked),'observations':data['coverage']['observations']}))


if __name__=='__main__':
    check(sys.argv[1])
