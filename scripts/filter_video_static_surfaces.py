"""Retain reconstructed surfaces supported by multiple unmasked depth exposures.

Free-space contradictions reject floating TSDF fragments. Occlusion is not a
contradiction, and person-masked pixels provide neither positive nor negative evidence.
"""
import argparse, copy, json, os, time
from pathlib import Path
import cv2
import numpy as np
from reconstruct_room_rgb import digest


def depth_evidence(points, depth, excluded, k, c2w, tolerance=.04):
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    projected = local @ k.T
    uv = np.rint(projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)).astype(np.int64)
    h, w = depth.shape
    inside = (local[:, 2] >= .2) & (local[:, 2] <= 5) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    indices = np.flatnonzero(inside); x, y = uv[inside].T
    z = depth[y, x]; valid = (z >= .2) & (z <= 5) & ~excluded[y, x]
    residual = z - local[indices, 2]
    return indices[valid & (abs(residual) <= tolerance)], indices[valid & (residual > tolerance)]


def self_check():
    depth = np.full((5, 5), 2.); blocked = np.zeros_like(depth, dtype=bool)
    k = np.array([[1., 0, 2], [0, 1, 2], [0, 0, 1]])
    points = np.array([[0, 0, 2.], [0, 0, 1.], [0, 0, 3.]])
    positive, negative = depth_evidence(points, depth, blocked, k, np.eye(4))
    assert positive.tolist() == [0] and negative.tolist() == [1] # behind observed depth is occluded
    blocked[2, 2] = True
    assert all(len(a) == 0 for a in depth_evidence(points, depth, blocked, k, np.eye(4)))
    print('PASS: static agreement, free-space contradictions, occlusion and dynamic-mask exclusion')


def run(args):
    import trimesh
    started = time.monotonic(); s = json.loads(args.scene.read_text()); n = json.loads(args.native_scene.read_text())
    a = json.loads(args.analysis.read_text()); af = {f['sourceFrame']: f for f in a['frames']}
    if s['provenance']['native_scene_sha256'] != digest(args.native_scene) or s['provenance']['analysis_sha256'] != digest(args.analysis):raise ValueError('Source hash differs')
    inputs = {f['source_index']: f for f in json.loads((Path(n['source_run'])/'input.manifest.json').read_text())}
    mesh = trimesh.load(args.scene.parent / s['meshUrl'], force='mesh', process=False)
    cloud = trimesh.load(args.scene.parent / s['pointCloudUrl'], force='scene', process=False).to_geometry()
    triangles=mesh.vertices[mesh.faces]
    texture_samples=np.concatenate((triangles, (triangles+np.roll(triangles,1,axis=1))/2,triangles.mean(1)[:,None]),axis=1)
    texture_frame=np.full(len(mesh.faces),-1,np.int32);texture_score=np.zeros(len(mesh.faces));texture_sources={}
    points = np.concatenate((mesh.vertices, cloud.vertices)); positive=np.zeros(len(points),np.int32); negative=positive.copy()
    k=np.array(s['provenance']['k']); count=0
    for f in s['frames']:
        index=f['sourceFrame']
        if index % 5 or index not in af or not af[index]['objects']:continue
        source=inputs[index]
        if digest(source['depth_source_path'])!=source['depth_sha256']:raise ValueError('Depth changed')
        depth=cv2.imread(source['depth_source_path'],-1)/s['provenance']['depth_factor']; excluded=np.zeros(depth.shape,np.uint8)
        for o in af[index]['objects']:
            path=(args.analysis.parent/o['maskUrl']).resolve()
            if not path.is_relative_to(args.analysis.parent.resolve()):raise ValueError('Mask outside analysis')
            excluded|=(cv2.imread(str(path),-1)[:,:,3]>0).astype(np.uint8)
        excluded=cv2.dilate(excluded,np.ones((5,5),np.uint8))>0
        pos,neg=depth_evidence(points,depth,excluded,k,np.array(f['c2w']))
        positive[pos]+=1;negative[neg]+=1;count+=1
        if index%30==0:
            c=np.array(f['c2w']);flat=texture_samples.reshape(-1,3)
            supported,_=depth_evidence(flat,depth,excluded,k,c)
            valid=np.zeros(len(flat),bool);valid[supported]=True;valid=valid.reshape(-1,7).all(1)
            local=(triangles-c[:3,3])@c[:3,:3];q=local@k.T;uv=q[:,:,:2]/np.maximum(q[:,:,2:],1e-8)
            edges=uv[:,1:]-uv[:,:1];area=abs(edges[:,0,0]*edges[:,1,1]-edges[:,0,1]*edges[:,1,0])/2
            better=valid&(area>texture_score)
            texture_frame[better]=index;texture_score[better]=area[better];texture_sources[index]=(source,c)

    keep=(positive>=3)&(negative<=np.maximum(2,positive*.15))
    mesh_keep=keep[:len(mesh.vertices)];face_keep=mesh_keep[mesh.faces].all(1)
    mesh.update_faces(face_keep);mesh.remove_unreferenced_vertices();texture_frame=texture_frame[face_keep]
    cloud_keep=keep[len(mesh_keep):];cloud=trimesh.points.PointCloud(cloud.vertices[cloud_keep],colors=cloud.colors[cloud_keep])
    if not len(mesh.faces) or not len(cloud.vertices):raise ValueError('No consistent static surfaces; keep original artifacts')
    args.output.mkdir(parents=True,exist_ok=False)
    from PIL import Image
    textured=trimesh.Scene()
    for index in np.unique(texture_frame):
        sub=mesh.submesh([np.flatnonzero(texture_frame==index)],append=True)
        if index>=0:
            source,c=texture_sources[int(index)]
            if digest(source['source_path'])!=source['sha256']:raise ValueError('Texture source changed')
            rgb=cv2.cvtColor(cv2.imread(source['source_path']),cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
            local=(sub.vertices-c[:3,3])@c[:3,:3];q=local@k.T;uv=(q[:,:2]/q[:,2:]+.5)/[w,h];uv[:,1]=1-uv[:,1]
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(rgb),metallicFactor=0,roughnessFactor=1,doubleSided=True)
            sub.visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=material)
        textured.add_geometry(sub,node_name=f'view-{index}')
    textured.export(args.output/'static-scene.glb');trimesh.Scene(cloud).export(args.output/'dense-static.glb');cloud.export(args.output/'dense-static.ply')
    result=copy.deepcopy(s)
    def rel(url):return os.path.relpath((args.scene.parent/url).resolve(),args.output.resolve())
    for o in result.get('staticObjects',[]):
        for key in ['meshUrl','provenanceUrl']:o[key]=rel(o[key])
        for key in ['maskUrl','imageUrl']:o['source'][key]=rel(o['source'][key])
    for f in result['frames']:
        for o in f['objects']:
            if o.get('surface'):o['surface']['meshUrl']=rel(o['surface']['meshUrl'])
    if result.get('semanticReview'):result['semanticReview']['url']=rel(result['semanticReview']['url'])
    for b in result.get('bodyKeyframes',[]):b['meshUrl']=rel(b['meshUrl'])
    result['meshUrl']='static-scene.glb';result['pointCloudUrl']='dense-static.glb';result['pointCloudCount']=len(cloud.vertices)
    report={'source_scene_sha256':digest(args.scene),'support_frames':count,'minimum_agreeing_frames':3,'depth_tolerance_m':.04,
            'maximum_free_space_contradictions':'max(2, 0.15 * positive_support)', 'input_triangles':len(face_keep),'retained_triangles':len(mesh.faces),
            'input_points':len(cloud_keep),'retained_points':len(cloud.vertices),'elapsedSeconds':time.monotonic()-started,
            'texture_mapped_triangles':int((texture_frame>=0).sum()),'texture_source_frames':sorted(int(x) for x in np.unique(texture_frame) if x>=0),
            'mesh_sha256':digest(args.output/'static-scene.glb'),'cloud_sha256':digest(args.output/'dense-static.glb')}
    result['provenance']['static_consistency']=report
    result['limitations'].append('静态网格与点云只保留至少3个未遮挡深度帧支持的区域；未观测区域仍不补造。')
    (args.output/'scene.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False));(args.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true')
    for name in ['scene','native-scene','analysis','output']:p.add_argument('--'+name,type=Path)
    args=p.parse_args();self_check() if args.self_check else run(args)
