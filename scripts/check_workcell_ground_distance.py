"""Small synthetic check: observed lower support, floor datum and view coverage."""
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workcell_photo_report import _ground_distance

def observation(photo, bottom):
    corners=[[x,y,z] for x in (-1,1) for y in (-2,2) for z in (bottom,bottom+1)]
    return {'photo':photo,'observedMeasurements':{'status':'available','basis':{'corners_native':corners}}}

geometry={'floor':{'normal':[0,0,2],'offset':-4},'clearances':[]}
transform=np.eye(4);transform[2,3]=-2
item={'id':'button','observations':[observation(1,2.3),observation(2,2.4)]}
result=_ground_distance(item,geometry,transform)
assert np.allclose(result['rangeNative'],[.3,.4])
for view in result['byPhoto'].values():
    assert abs(view['footNative'][2])<1e-9
    assert np.isclose(np.linalg.norm(np.array(view['pointNative'])-view['footNative']),view['valueNative'])
assert not _ground_distance({**item,'observations':item['observations'][:1]},geometry,transform)['byPhoto']
assert not _ground_distance({**item,'observations':[observation(1,2.3),observation(1,2.4)]},geometry,transform)['byPhoto']
assert _ground_distance({**item,'observations':[observation(1,1.9),observation(2,2.4)]},geometry,transform)['byPhoto']['1']['valueNative'] is None
bad={'floor':{'normal':[0,0,0],'offset':0},'clearances':[]}
assert not _ground_distance(item,bad,transform)['byPhoto']
invalid=observation(2,2.4);invalid['observedMeasurements']['basis']['corners_native']=[]
assert not _ground_distance({**item,'observations':[observation(1,2.3),invalid]},geometry,transform)['byPhoto']
negative_rail={**geometry,'clearances':[{'id':'fence-plane-0-lower-rail','heightNative':-.1,'pointNative':[0,0,1.9],'footNative':[0,0,2],'sourcePhotos':[1,2]}]}
assert _ground_distance({**item,'id':'fence-0'},negative_rail,transform)['feature']['valueNative'] is None
rotation=trimesh.transformations.rotation_matrix(.6,[1,0,0]); rotated=[]
for obs in item['observations']:
    corners=trimesh.transform_points(obs['observedMeasurements']['basis']['corners_native'],rotation)
    rotated.append({'photo':obs['photo'],'observedMeasurements':{'status':'available','basis':{'corners_native':corners.tolist()}}})
tilted={'floor':{'normal':(rotation[:3,:3]@np.array([0,0,2])).tolist(),'offset':-4},'clearances':[]}
assert np.allclose(_ground_distance({**item,'observations':rotated},tilted,transform@np.linalg.inv(rotation))['rangeNative'],[.3,.4])
print('PASS: floor normalization, signed support, rigid invariance, multi-view gate and matching annotation endpoints')
