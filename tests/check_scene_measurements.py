"""Analytic regression checks; no models, GPU or network required."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ehs_spatial.platform.scene_measurements import fitted_plane, surface_distance, measure_scene, register_measurement_routes
from ehs_spatial.platform.contracts import PlatformError

square=np.array([[[0,0,0],[2,0,0],[2,2,0]],[[0,0,0],[2,2,0],[0,2,0]]],float)
p=fitted_plane(square)
assert abs(p['normal'][2])>.99999
rot=np.array([[1,0,0],[0,.5,-np.sqrt(.75)],[0,np.sqrt(.75),.5]])
q=fitted_plane(square@rot.T)
assert abs(np.degrees(np.arccos(abs(p['normal']@q['normal'])))-60)<1e-6
# Subdivision should not alter area-weighted fit.
assert np.allclose(p['center'],[1,1,0])
for other,expected in [(square+[0,0,3],3),(square+[3,0,0],1),(square+.25, .25), (square@rot.T,0)]:
 value,(a,b)=surface_distance(square,other)
 assert abs(value-expected)<1e-8,(value,expected)
 assert abs(np.linalg.norm(a-b)-value)<1e-8
# Face/edge intersection with all endpoints outside the other triangle.
vertical=np.array([[[.5,-1,-1],[.5,3,-1],[.5,1,2]]])
assert surface_distance(square,vertical)[0]==0
# Vertex-only nearest-neighbor would incorrectly overestimate this separation.
small=np.array([[[.9,.9,1],[1.1,.9,1],[1,1.1,1]]])
assert abs(surface_distance(square,small)[0]-1)<1e-9

def entity(id,position,quaternion=[0,0,0,1],scale=[1,1,1]):
 return {'id':id,'activeModelRepresentationId':id+'-rep','representations':[{'id':id+'-rep','kind':'primitive','coordinateFrameId':'frame','primitive':{'type':'box','dimensions':[2,2,.02]},'transform':{'coordinateFrameId':'frame','position':position,'quaternion':quaternion,'scale':scale}}]}
a,b=entity('a',[0,0,0]),entity('b',[0,0,3])
revision={'id':'revision','document':{'entities':[a,b],'assets':[],'coordinateFrames':[{'id':'frame','ground':{'normal':[0,0,1]}}]}}
load=lambda _: (_ for _ in ()).throw(AssertionError('Unexpected asset load'))
r=measure_scene(revision,'distance','a','b',None,load)
assert abs(r['value']-2.98)<1e-8 and r['unit']=='native'
r=measure_scene(revision,'angle','a','b',None,load)
assert r['value']<1e-6 and len(r['lines'])==2
region={'coordinateFrameId':'frame','nativeToPlane':np.eye(4).tolist(),'points':[[0,0],[2,0],[2,2],[0,2]]}
r=measure_scene(revision,'occupancy','a',None,region,load)
assert abs(r['value']-1)<1e-8 and r['unit']=='native2'
assert r['quality']['regionFraction']==.25
for change,code in [(lambda: b['representations'][0]['transform'].update(coordinateFrameId='other'),'measurement_frame_mismatch')]:
 change()
 try:measure_scene(revision,'angle','a','b',None,load)
 except PlatformError as e:assert e.code==code
 else:raise AssertionError(code)
b['representations'][0]['transform']['coordinateFrameId']='frame'
# Nonuniform scale must affect the surface normal, not merely quaternion axes.
b['representations'][0]['transform'].update(quaternion=[float(np.sin(np.pi/6)),0,0,float(np.cos(np.pi/6))],scale=[2,1,3])
assert abs(measure_scene(revision,'angle','a','b',None,load)['value']-60)<1e-6
try:measure_scene(revision,'occupancy','a',None,{**region,'points':[[0,0],[1,1],[1,0],[0,1]]},load)
except PlatformError as e:assert e.code=='measurement_region_invalid'
else:raise AssertionError('self-intersecting region')
app=FastAPI();register_measurement_routes(app,lambda _:revision,load)
with TestClient(app) as client:
 assert client.get('/api/revisions/revision/measurements',params={'kind':'angle','entityA':'a','entityB':'b'}).status_code==200
print('PASS: planes, true mesh-surface distance, crossings, interior closest points, region area, invalid boundaries, frames and HTTP calculation')

# Mesh holes stay empty under projection, unlike an enclosing outline/box.
from copy import deepcopy
from ehs_spatial.platform.spatial import MeshData
before=deepcopy(revision)
ring=[]
for x0,y0,x1,y1 in [(0,0,3,1),(0,2,3,3),(0,1,1,2),(2,1,3,2)]:
 ring.extend([[[x0,y0,0],[x1,y0,0],[x1,y1,0]],[[x0,y0,0],[x1,y1,0],[x0,y1,0]]])
mesh=MeshData(np.asarray(ring).reshape(-1,3),np.arange(24).reshape(-1,3))
r=deepcopy(revision); rep=r['document']['entities'][0]['representations'][0]
rep.update(kind='generated_mesh',assetId='mesh'); r['document']['assets']=[{'id':'mesh'}]
hole={**region,'points':[[1,1],[2,1],[2,2],[1,2]]}
assert measure_scene(r,'occupancy','a',None,hole,lambda _:mesh)['value']==0
assert revision==before
# An oblique surface under anisotropic scale uses its transformed normal.
tilt=square@rot.T*np.array([2,1,3]); normal=np.array([0,-np.sqrt(.75),.5])/np.array([2,1,3]); normal/=np.linalg.norm(normal)
assert abs(fitted_plane(tilt)['normal']@normal)>1-1e-8
for field,value,code in [('visible',False,'measurement_object_missing'),('sourceContext',True,'measurement_object_missing')]:
 broken=deepcopy(revision);broken['document']['entities'][0][field]=value
 try:measure_scene(broken,'angle','a','b',None,load)
 except PlatformError as e:assert e.code==code
 else:raise AssertionError(code)
broken=deepcopy(revision);broken['document']['entities'][0]['representations'][0]['sourceValidity']='stale'
try:measure_scene(broken,'angle','a','b',None,load)
except PlatformError as e:assert e.code=='measurement_model_missing'
else:raise AssertionError('stale model accepted')
broken=deepcopy(revision);broken['document']['coordinateFrames'][0].pop('ground')
try:measure_scene(broken,'occupancy','a',None,region,load)
except PlatformError as e:assert e.code=='measurement_ground_missing'
else:raise AssertionError('missing ground accepted')
print('PASS: projection holes, immutable input, anisotropic normals, hidden/excluded/stale rejection and ground qualification')
