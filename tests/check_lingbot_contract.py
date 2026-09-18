"""Run directly: verifies camera convention, raster shape and stream source ordering."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from modal_apps import lingbot_room as contract
from unittest.mock import patch
from types import SimpleNamespace
with patch.object(contract.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)):
 contract.require_disk_space(Path.cwd())
 try:contract.require_disk_space(Path.cwd(),1)
 except OSError:pass
 else:raise AssertionError('A job must retain disk reserve after its expected output')
with patch.object(contract.shutil, 'disk_usage', return_value=SimpleNamespace(free=700*1024**2)):
 try:contract.require_disk_space(Path.cwd())
 except OSError:pass
 else:raise AssertionError('Low disk space must reject new local artifacts')
k=np.array([[100.,0,2],[0,100.,1],[0,0,1]])
c=np.eye(4);c[:3,3]=[1,2,3]
y,x=np.indices((3,5));z=np.full((3,5),2.)
points=(np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(k).T)*z[...,None]+c[:3,3]
r=contract.validate_prediction(z,np.ones_like(z),points,k,c)
assert r['pointmap_xyz_reprojection_error_p95_px']<1e-6
assert r['pointmap_camera_z_residual_p95']<1e-6
for bad in [np.linalg.inv(c), np.zeros((4,4))]:
 try:contract.validate_prediction(z,np.ones_like(z),points,k,bad)
 except ValueError:pass
 else:raise AssertionError('Wrong camera convention must fail')
for indices in [[0,3,3],[3,0,6],[-1,3,6]]:
 try:contract.validate_indices(indices,8)
 except ValueError:pass
 else:raise AssertionError('Invalid source frame sequence must fail')
assert contract.validate_indices([0,3,6],8)==[0,3,6]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_replay_scene import world_points
c[:3,:3]=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
xyz=world_points(np.column_stack([x.ravel(),y.ravel()]),z.ravel(),k,c).reshape(*z.shape,3)
r=contract.validate_prediction(z,np.ones_like(z),xyz,k,c)
assert r['pointmap_xyz_reprojection_error_p95_px']<1e-6
assert r['pointmap_camera_z_residual_p95']<1e-6
from build_video_object_models import observed_surface
color=np.zeros((3,5,3),np.uint8);mask=np.ones_like(z,bool)
v,f,*_=observed_surface(color,z,mask,k,c)
v2,f2,*_=observed_surface(color,z*10,mask,k,c,max_edge_m=.5,depth_range=(0,np.inf))
assert len(f)>0 and np.array_equal(f,f2)
assert np.allclose(v2-c[:3,3],10*(v-c[:3,3])),'Native units must not inherit metric depth clipping'
from build_lingbot_replay import source_transform,resize_mask
from build_lingbot_replay import read_prediction
from tempfile import TemporaryDirectory
with TemporaryDirectory() as temp:
 path=Path(temp)/'native.npz';expected_camera=c.copy()
 np.savez(path,depth=z[:,:,None],depth_conf=np.ones_like(z),rgb=color,k=k,w2c=np.linalg.inv(c)[:3])
 *_,actual_camera,valid=read_prediction(path)
 assert np.allclose(actual_camera,expected_camera),'Official saved W2C must be inverted exactly once'
from build_video_body_models import camera_body
from PIL import Image
for shape in [(480,640),(900,600)]:
 transform,rh,y0=source_transform(shape)
 mask=np.zeros(shape,np.uint8);mask[shape[0]//3:shape[0]*2//3,shape[1]//4:shape[1]*3//4]=1
 expected=np.asarray(Image.fromarray(mask).resize((518,rh),Image.Resampling.NEAREST))[y0:y0+min(rh,518)]>0
 assert np.array_equal(resize_mask(mask,shape),expected),'Mask centers must match official RGB resize/crop'
rng=np.random.default_rng(17);joints=rng.normal(size=(70,3))*.15
k=np.array([[500.,0,320],[0,500,240],[0,0,1.]])
camera=joints+[0,0,3];uv=camera@k.T;pixels=uv[:,:2]/uv[:,2:]
person={'keypoints_3d':joints.tolist(),'keypoints_2d':pixels.tolist(),'pred_cam_t':[0,0,3]}
mesh=SimpleNamespace(vertices=camera*[1,-1,-1])
transform,_,_=source_transform((480,640));resized_k=transform@k
vertices,camera_joints,error=camera_body(mesh,person,resized_k,transform)
assert np.max(error)<1e-6 and np.allclose(vertices,camera,atol=1e-6) and np.allclose(camera_joints,camera,atol=1e-6),'Cached body must use resized camera pixel coordinates'
print('PASS: LingBot camera/depth/XYZ and chronological source contract')
