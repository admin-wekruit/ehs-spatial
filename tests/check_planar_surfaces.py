"""Run directly: analytic local-surface geometry and persisted report contract."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from ehs_spatial.platform.planar_surfaces import extract_planar_surfaces, surface_inclinations

CONFIG={'minAreaNative2':.02,'distanceToleranceNative':.003,'normalToleranceDeg':10.}
def panel(deg, width=1., length=2., offset=(0,0,0), splits=1):
 angle=np.deg2rad(deg); d=np.array([0,np.cos(angle),np.sin(angle)])
 grid=np.array([[np.array([x,0,0])+d*y+offset for y in np.linspace(0,width,splits+1)] for x in np.linspace(0,length,splits+1)])
 return np.array([grid[[i,i+1,i+1],[j,j,j+1]] for i in range(splits) for j in range(splits)]+[grid[[i,i+1,i],[j,j+1,j+1]] for i in range(splits) for j in range(splits)])
ground={'normal':[0,0,1],'angularErrorDeg':0.}
for deg in (0,30,60,90):
 for splits in (1,8):
  surfaces=extract_planar_surfaces(panel(deg,splits=splits),CONFIG)
  assert len(surfaces)==1,(deg,splits,len(surfaces))
  assert abs(surfaces[0]['areaNative2']-2)<1e-6
  rows=surface_inclinations(surfaces,ground)
  assert abs(rows[0]['inclinationDeg']-deg)<.1
multi=np.concatenate([panel(0,offset=(0,0,0)),panel(30,offset=(4,0,0)),panel(60,offset=(8,0,0))])
surfaces=extract_planar_surfaces(multi,CONFIG)
assert len(surfaces)==3
assert sorted(round(s['inclinationDeg']) for s in surface_inclinations(surfaces,ground))==[0,30,60]
# Coplanar but disconnected patches stay separate.
assert len(extract_planar_surfaces(np.concatenate([panel(30),panel(30,offset=(4,0,0))]),CONFIG))==2
thin=panel(60,width=.06);n=np.cross(thin[0,1]-thin[0,0],thin[0,2]-thin[0,0]);n/=np.linalg.norm(n)
solid=np.concatenate([thin+n*.001,(thin-n*.001)[:,::-1]])
assert len(extract_planar_surfaces(solid,CONFIG))==1
assert abs(extract_planar_surfaces(solid,CONFIG)[0]['areaNative2']-.12)<1e-6
assert not extract_planar_surfaces(panel(30,width=.001,length=.001),CONFIG)
angle=.72;rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
turned=extract_planar_surfaces(multi@rot.T+[4,-3,2],CONFIG)
assert sorted(round(s['inclinationDeg']) for s in surface_inclinations(turned,{'normal':rot@np.array([0.,0,1]),'angularErrorDeg':0.}))==[0,30,60]
assert len(extract_planar_surfaces(solid[:,::-1],CONFIG))==1
for g in ({},{'normal':[0,0,0]},{'normal':[float('nan'),0,1]}):
 try:surface_inclinations(surfaces,g)
 except ValueError:pass
 else:raise AssertionError('Missing/invalid ground cannot yield an inclination')
print('PASS: multi-plane angles, local area, narrow sheet, thickness, disconnected patches, re-tessellation, winding, pose and invalid ground')
# Curved geometry does not become a set of arbitrary planar labels.
import open3d as o3d
sphere=o3d.geometry.TriangleMesh.create_sphere(radius=1,resolution=30)
assert not extract_planar_surfaces(np.asarray(sphere.vertices)[np.asarray(sphere.triangles)],CONFIG)
# Projection preserves a square hole rather than filling its convex hull.
ring=np.concatenate([panel(0,width=.25,length=2),panel(0,width=.25,length=2,offset=(0,1.75,0)),panel(0,width=1.5,length=.25,offset=(0,.25,0)),panel(0,width=1.5,length=.25,offset=(1.75,.25,0))])
r=extract_planar_surfaces(ring,CONFIG)
assert len(r)==1 and len(r[0]['boundary'])==2 and abs(r[0]['areaNative2']-1.75)<1e-6
# A faceted cylinder has genuinely flat narrow strips. Their inclination must
# stay vertical; mesh-only processing cannot infer a smooth CAD cylinder.
cylinder=o3d.geometry.TriangleMesh.create_cylinder(radius=1,height=2,resolution=60,split=4)
ct=np.asarray(cylinder.vertices)[np.asarray(cylinder.triangles)]
cn=np.cross(ct[:,1]-ct[:,0],ct[:,2]-ct[:,0]);ct=ct[abs(cn[:,2])<1e-10]
cylinder_patches=extract_planar_surfaces(ct,CONFIG)
assert all(abs(s['inclinationDeg']-90)<.1 for s in surface_inclinations(cylinder_patches,ground))
# Parameter comparison: small positional noise survives; tiny patches do not.
noisy=panel(30,splits=8)+np.random.default_rng(2).normal(0,.0001,(128,3,3))
for area in (.01,.02,.05):
 for distance in (.003,.01):
  cfg={**CONFIG,'minAreaNative2':area,'distanceToleranceNative':distance}
  assert len(extract_planar_surfaces(solid,cfg))==1
  assert not extract_planar_surfaces(panel(30,width=.001,length=.001),cfg)
  patches=extract_planar_surfaces(noisy,cfg)
  assert patches and abs(surface_inclinations(patches,ground)[0]['inclinationDeg']-30)<.1
print('PASS: curved rejection, actual supported holes, noise and engineering-parameter comparison')

from copy import deepcopy
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ehs_spatial.platform.spatial import MeshData
from ehs_spatial.platform.scene_measurements import analyze_inclinations,saved_inclinations,register_measurement_routes
entity={'id':'object','activeModelRepresentationId':'rep','representations':[{'id':'rep','kind':'generated_mesh','assetId':'mesh','coordinateFrameId':'frame','transform':{'coordinateFrameId':'frame','position':[0,0,0],'quaternion':[0,0,0,1],'scale':[1,1,1]}}]}
revision={'id':'revision','document':{'entities':[entity,{'id':'hidden','visible':False}],'assets':[{'id':'mesh','sha256':'bound-hash'}],'coordinateFrames':[{'id':'frame','ground':ground}]}}
mesh=MeshData(multi.reshape(-1,3),np.arange(len(multi)*3).reshape(-1,3))
calls=[]
def load(identity):calls.append(identity);return mesh
original=deepcopy(revision);cache={}
a=analyze_inclinations(revision,load,cache=cache,config=CONFIG)
assert revision==original and len(a['items'])==1
assert len(a['items'][0]['surfaces'])==3 and a['items'][0]['status']=='measured'
assert all(s['result']['references'][0]['entityId']=='object' for s in a['items'][0]['surfaces'])
count=len(calls);analyze_inclinations(revision,load,cache=cache,config=CONFIG);assert len(calls)==count
analyze_inclinations(revision,load,persist=True,config=CONFIG)
assert saved_inclinations(revision,CONFIG)['items'][0]['status']=='measured'
assert saved_inclinations(revision,{**CONFIG,'minAreaNative2':.03})['items'][0]['status']=='not_processed'
revision['document']['coordinateFrames'][0]['ground']['normal']=[0,1,0]
assert saved_inclinations(revision,CONFIG)['items'][0]['status']=='not_processed'
revision['document']['coordinateFrames'][0]['ground']=ground.copy()
revision['document']['entities'][0]['representations'][0]['transform']['position']=[1,0,0]
assert saved_inclinations(revision,CONFIG)['items'][0]['status']=='not_processed'
assert saved_inclinations(original,{**CONFIG,'minAreaNative2':.03})['items'][0]['status']=='not_processed'
unknown=surface_inclinations(surfaces,{'normal':[0,0,1]})
assert all(s['classification']=='direction_unverified' and s['angularErrorDeg'] is None for s in unknown)
missing=deepcopy(original);missing['document']['coordinateFrames'][0]['ground']=None
assert analyze_inclinations(missing,load)['items'][0]['reason']=='measurement_ground_missing'
app=FastAPI()
register_measurement_routes(app,lambda _: (_ for _ in ()).throw(AssertionError('Full revision read')),lambda _: (_ for _ in ()).throw(AssertionError('Mesh read')),get_inclinations=lambda _:a)
with TestClient(app) as client:
 assert client.get('/api/revisions/revision/inclination-analysis-v1').json()==a
print('PASS: exact-input caching, immutable source, ground/pose invalidation, unknown ground accuracy, compute-free HTTP')

# Exhausting the detector without a surface is incomplete, not a negative finding.
def exhausted(*args,diagnostics,**kwargs):
 diagnostics['limitReached']=True
 return []
with patch('ehs_spatial.platform.planar_surfaces.extract_planar_surfaces',side_effect=exhausted):
 limited=analyze_inclinations(original,load,config=CONFIG)['items'][0]
 assert limited['status']=='partial' and limited['reason']=='measurement_complexity_limit'
# A retried failed row must not retain its obsolete error after success.
retry=deepcopy(original)
retry['document']['entities'][0]['inclinationAnalysis']={**a['items'][0],'status':'failed','reason':'asset_not_found'}
recovered=analyze_inclinations(retry,load,config=CONFIG)['items'][0]
assert recovered['status']=='measured' and 'reason' not in recovered
print('PASS: incomplete detector outcomes and retry error clearing')

# Interactive requests retain their resource guard; offline processing must not
# reject an otherwise valid model merely for having many (including empty) faces.
from ehs_spatial.platform.scene_measurements import measure_scene
from ehs_spatial.platform.contracts import PlatformError
large_mesh=MeshData(np.array([[0,0,0],[2,0,0],[2,1,0],[0,1,0]],float),np.vstack([[[0,1,2],[0,2,3]],np.zeros((500000,3),dtype=int)]))
try:measure_scene(original,'inclination','object',None,None,lambda _:large_mesh)
except PlatformError as exc:assert exc.code=='measurement_complexity_limit'
else:raise AssertionError('Interactive work must remain bounded')
large=analyze_inclinations(original,lambda _:large_mesh,config=CONFIG)['items'][0]
assert large['status']=='measured' and len(large['surfaces'])==1,large
print('PASS: offline large models and interactive resource guard')
from ehs_spatial.platform.scene_measurements import analyze_bends
large_bend=analyze_bends(original,lambda _:large_mesh)['items'][0]
assert large_bend['status']=='unsupported' and large_bend['reason']=='measurement_no_stable_bend',large_bend
print('PASS: offline bend batch also processes full meshes')
