"""Check exported world-space assets against their native source, without refitting.

Run with --scene, --run, --analysis and --output paths. No model calls.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_lingbot_replay import read_prediction,resize_mask
from reconstruct_room_rgb import digest

p=argparse.ArgumentParser(description=__doc__)
for name in ['scene','run','analysis','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();scene=json.loads(a.scene.read_text());execution=json.loads((a.run/'run.json').read_text())
analysis=json.loads(a.analysis.read_text());observations={f['sourceFrame']:f for f in analysis['frames']}
assert scene['provenance']['execution_sha256']==digest(a.run/'run.json')
assert scene['provenance']['analysis_sha256']==digest(a.analysis)
assert scene['provenance']['temporal_geometry_check']['passed']
assert scene['source_video_sha256']==analysis['provenance']['sourceVideoSha256']
review=scene['provenance']['static_consistency']
for url,key in [('meshUrl','mesh_sha256'),('pointCloudUrl','cloud_sha256')]:assert digest(a.scene.parent/scene[url])==review[key]
bodies={(b['sourceFrame'],b['entityId']):b for b in scene['bodyKeyframes']};records=[];surfaces=0
for frame,native in zip(scene['frames'],execution['frames'],strict=True):
    index=frame['sourceFrame'];assert index==native['sourceFrame']
    path=a.run/native['file'];assert digest(path)==native['sha256']
    depth,confidence,rgb,k,camera,valid=read_prediction(path)
    assert np.allclose(camera,frame['c2w'],atol=1e-7)
    original={o['entityId']:o for o in observations[index]['objects']}
    for obj in frame['objects']:
        rgba=cv2.imread(str(a.analysis.parent/original[obj['entityId']]['maskUrl']),-1)
        mask=resize_mask(rgba[:,:,3]>0,(analysis['height'],analysis['width']))
        if obj.get('surface'):
            surface=obj['surface'];path=a.scene.parent/surface['meshUrl'];assert digest(path)==surface['sha256']
            mesh=trimesh.load(path,force='mesh',process=False)
            local=(mesh.vertices-camera[:3,3])@camera[:3,:3];q=local@k.T
            uv=q[:,:2]/q[:,2:];pixels=np.rint(uv).astype(int)
            assert np.max(abs(uv-pixels))<.001 and (local[:,2]>0).all()
            assert mask[pixels[:,1],pixels[:,0]].all();surfaces+=1
        b=bodies.get((index,obj['entityId']))
        if not b:continue
        path=a.scene.parent/b['meshUrl'];assert digest(path)==b['mesh_sha256']
        mesh=trimesh.load(path,force='mesh',process=False)
        local=(mesh.vertices-camera[:3,3])@camera[:3,:3]
        cast=o3d.t.geometry.RaycastingScene()
        cast.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(local.astype('float32')),o3d.core.Tensor(mesh.faces.astype('uint32'))))
        h,w=depth.shape;z=cast.cast_rays(cast.create_rays_pinhole(k,np.eye(4),w,h))['t_hit'].numpy()
        visible=np.isfinite(z)&(z>0);overlap=visible&mask&valid&(confidence>=1.5)
        iou=float((visible&mask).sum()/max(1,(visible|mask).sum()))
        residual=abs(z[overlap]-depth[overlap])/depth[overlap]
        assert overlap.sum()>=300 and iou>=.65 and np.median(residual)<=.04001 and np.percentile(residual,95)<=.10001
        records.append({'sourceFrame':index,'entityId':obj['entityId'],'iou':iou,'relative_depth_p95':float(np.percentile(residual,95))})
for obj in scene['staticObjects']:
    source=obj['source'];original=observations[source['sourceFrame']]
    assert source['timeSec']==original['timeSec'] and source['endTimeSec']==original['endTimeSec'],'Held 3D interval must not widen the original mask interval'
    proof=json.loads((a.scene.parent/obj['provenanceUrl']).read_text())
    for path,key in [(obj['meshUrl'],'mesh_sha256'),(source['maskUrl'],'source_mask_sha256'),(source['imageUrl'],'source_image_sha256')]:assert digest(a.scene.parent/path)==proof[key]
assert len(records)==len(bodies) and surfaces==scene['humanSurfaces']['count']
report={'scene_sha256':digest(a.scene),'exported_bodies_checked':len(records),'exported_surfaces_checked':surfaces,
        'object_source_intervals_checked':len(scene['staticObjects']),'refitted_during_check':False,'field_accuracy_validated':False,'bodies':records}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='bodies'}))
