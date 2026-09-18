"""One known tilted-surface check: improve depth while retaining source pixels."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from attach_lingbot_objects import refine_body_alignment
k=np.array([[200.,0,80],[0,200.,60],[0,0,1.]])
y,x=np.indices((120,160));mask=(x>30)&(x<130)&(y>20)&(y<100)
z=2+.12*(x-80)/80;depth=np.full_like(z,2.)
pixels=np.column_stack((x[mask]+.5,y[mask]+.5))
vertices=np.column_stack((pixels,np.ones(len(pixels))))@np.linalg.inv(k).T*z[mask,None]
joints=vertices[np.linspace(0,len(vertices)-1,70,dtype=int)]
q=joints@k.T;joint_pixels=q[:,:2]/q[:,2:]
v,j,report=refine_body_alignment(vertices,joints,joint_pixels,k,z,depth,mask)
assert report['converged'] and report['final_cost']<report['initial_cost']
assert np.mean(abs(v[:,2]-2))<np.mean(abs(vertices[:,2]-2))*.3
q=j@k.T;assert np.percentile(np.linalg.norm(q[:,:2]/q[:,2:]-joint_pixels,axis=1),95)<5
scaled,scaled_joints,_=refine_body_alignment(vertices*8,joints*8,joint_pixels,k,z*8,depth*8,mask)
assert np.allclose(scaled/8,v,atol=1e-6) and np.allclose(scaled_joints/8,j,atol=1e-6),'No metric-scale assumptions'
matrix=np.array(report['camera_similarity'])
assert np.allclose(v,vertices@matrix[:3,:3].T+matrix[:3,3]),'Saved transform must reproduce the exported vertices'
singular=np.linalg.svd(matrix[:3,:3],compute_uv=False)
assert np.ptp(singular)<1e-10 and np.linalg.det(matrix[:3,:3])>0,'Do not distort or reflect anatomy'
print('PASS: constrained pose refinement improves depth, preserves pixels and native-scale similarity')
