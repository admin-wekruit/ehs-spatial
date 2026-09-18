"""Retain reconstructed surfaces supported by multiple unmasked depth exposures.

Free-space contradictions reject floating TSDF fragments. Occlusion is not a
contradiction, and person-masked pixels provide neither positive nor negative evidence.
"""
import argparse, copy, json, os, time
from pathlib import Path
import cv2
import numpy as np
from reconstruct_room_rgb import digest


def point_mesh(cloud, cameras, spacing):
    """Connect measured points locally; do not extrapolate a closed room."""
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree
    points = np.asarray(cloud.vertices)
    if len(points) < 30 or not np.isfinite(points).all() or not np.isfinite(cameras).all() or not 0 < spacing < 1:
        raise ValueError('Invalid point-mesh source')
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[:, :3] / 255)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=spacing*4, max_nn=30))
    _, nearest = cKDTree(cameras).query(points)
    normals = np.asarray(pc.normals)
    normals[(normals * (cameras[nearest] - points)).sum(1) < 0] *= -1
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector([spacing*r for r in (1.5, 2, 3)]))
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles(); mesh.remove_unreferenced_vertices()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
        vertex_colors=np.rint(np.asarray(mesh.vertex_colors)*255).astype('uint8'), process=False)


def depth_evidence(points, depth, excluded, k, c2w, tolerance=.04, *, depth_range=(.2,5), relative_tolerance=0):
    if not 0<=depth_range[0]<depth_range[1] or tolerance<0 or relative_tolerance<0:
        raise ValueError('Invalid depth evidence bounds')
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    projected = local @ k.T
    uv = np.rint(projected[:, :2] / np.maximum(projected[:, 2:], 1e-8)).astype(np.int64)
    h, w = depth.shape
    inside = (local[:, 2] > 0) & (local[:, 2] >= depth_range[0]) & (local[:, 2] <= depth_range[1]) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    indices = np.flatnonzero(inside); x, y = uv[inside].T
    z = depth[y, x]; valid = np.isfinite(z) & (z > 0) & (z >= depth_range[0]) & (z <= depth_range[1]) & ~excluded[y, x]
    residual = z - local[indices, 2]
    limit=np.maximum(tolerance,z*relative_tolerance)
    return indices[valid & (abs(residual) <= limit)], indices[valid & (residual > limit)]


def self_check():
    depth = np.full((5, 5), 2.); blocked = np.zeros_like(depth, dtype=bool)
    k = np.array([[1., 0, 2], [0, 1, 2], [0, 0, 1]])
    points = np.array([[0, 0, 2.], [0, 0, 1.], [0, 0, 3.]])
    positive, negative = depth_evidence(points, depth, blocked, k, np.eye(4))
    assert positive.tolist() == [0] and negative.tolist() == [1] # behind observed depth is occluded
    relative=depth_evidence(points*20,depth*20,blocked,k,np.eye(4),tolerance=0,depth_range=(0,np.inf),relative_tolerance=.04)
    assert relative[0].tolist()==[0] and relative[1].tolist()==[1], 'Monocular units must not inherit a 5m clipping range'
    blocked[2, 2] = True
    assert all(len(a) == 0 for a in depth_evidence(points, depth, blocked, k, np.eye(4)))
    import trimesh
    x, y = np.meshgrid(np.arange(12)*.015, np.arange(12)*.015)
    patches = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    patches = np.concatenate((patches, patches+[1,0,0]))
    surface = point_mesh(trimesh.points.PointCloud(patches, colors=np.tile([150,150,150,255], (len(patches),1))), np.array([[0,0,0]]), .015)
    assert len(surface.faces) > 200 and np.allclose(surface.vertices[:,2],1)
    assert np.max(np.linalg.norm(np.diff(surface.vertices[surface.faces],axis=1),axis=2)) < .1
    assert all(np.any(np.all(np.isclose(patches,v),axis=1)) for v in surface.vertices)
    print('PASS: static agreement, free-space contradictions, occlusion and dynamic-mask exclusion')
    print('PASS: point-based reconstruction retains source XYZ and never bridges separated surfaces')


def run(args):
    import trimesh
    started = time.monotonic(); s = json.loads(args.scene.read_text())
    a = json.loads(args.analysis.read_text()); af = {f['sourceFrame']: f for f in a['frames']}
    if s['provenance']['analysis_sha256'] != digest(args.analysis):raise ValueError('Mask source hash differs')
    predicted=bool(args.lingbot_run)
    if predicted and (args.native_scene or args.mesh_from_points):raise ValueError('Choose one geometry source and its supported reconstruction')
    if predicted:
        from build_lingbot_replay import read_prediction, resize_mask
        execution=json.loads((args.lingbot_run/'run.json').read_text())
        if s['provenance']['execution_sha256']!=digest(args.lingbot_run/'run.json') or s['units']!='uncalibrated_monocular':raise ValueError('Native prediction source differs')
        inputs={f['sourceFrame']:f for f in execution['frames']}
        evidence_options={'tolerance':0,'depth_range':(0,np.inf),'relative_tolerance':.04}
    else:
        n=json.loads(args.native_scene.read_text())
        if s['provenance']['native_scene_sha256']!=digest(args.native_scene):raise ValueError('Camera source hash differs')
        inputs = {f['source_index']: f for f in json.loads((Path(n['source_run'])/'input.manifest.json').read_text())}
        evidence_options={}
    cloud = trimesh.load(args.scene.parent / s['pointCloudUrl'], force='scene', process=False).to_geometry()
    if args.mesh_from_points:
        evidence=s['provenance'].get('static_consistency',{})
        if evidence.get('cloud_sha256') != digest(args.scene.parent/s['pointCloudUrl']): raise ValueError('Point mesh requires the checked static cloud')
        spacing=s['provenance']['dense_cloud']['voxel_m']
        mesh=point_mesh(cloud,np.array([np.array(f['c2w'])[:3,3] for f in s['frames']]),spacing)
    else: mesh = trimesh.load(args.scene.parent / s['meshUrl'], force='mesh', process=False)
    triangles=mesh.vertices[mesh.faces]
    texture_samples=np.concatenate((triangles, (triangles+np.roll(triangles,1,axis=1))/2,triangles.mean(1)[:,None]),axis=1)
    texture_frame=np.full(len(mesh.faces),-1,np.int32);texture_score=np.zeros(len(mesh.faces));texture_sources={}
    points = np.concatenate((mesh.vertices, cloud.vertices)); positive=np.zeros(len(points),np.int32); negative=positive.copy()
    count=0
    for f in s['frames']:
        index=f['sourceFrame']
        if index % (9 if predicted else 5) or index not in af or not af[index]['objects']:continue
        source=inputs[index]
        if predicted:
            path=args.lingbot_run/source['file']
            if digest(path)!=source['sha256']:raise ValueError('Prediction changed')
            depth,confidence,color,k,c,valid=read_prediction(path)
            if not np.allclose(c,f['c2w'],atol=1e-7):raise ValueError('Predicted camera differs from replay')
            source={**source,'color':color};excluded=(~valid|(confidence<1.5)).astype('uint8')
        else:
            if digest(source['depth_source_path'])!=source['depth_sha256']:raise ValueError('Depth changed')
            depth=cv2.imread(source['depth_source_path'],-1)/s['provenance']['depth_factor']; excluded=np.zeros(depth.shape,np.uint8)
            k=np.array(s['provenance']['k'])
        for o in af[index]['objects']:
            path=(args.analysis.parent/o['maskUrl']).resolve()
            if not path.is_relative_to(args.analysis.parent.resolve()):raise ValueError('Mask outside analysis')
            mask=cv2.imread(str(path),-1)[:,:,3]>0
            if predicted:mask=resize_mask(mask,(a['height'],a['width']))
            excluded|=mask.astype(np.uint8)
        excluded=cv2.dilate(excluded,np.ones((5,5),np.uint8))>0
        pos,neg=depth_evidence(points,depth,excluded,k,np.array(f['c2w']),**evidence_options)
        positive[pos]+=1;negative[neg]+=1;count+=1
        if index%30==0:
            c=np.array(f['c2w']);flat=texture_samples.reshape(-1,3)
            supported,_=depth_evidence(flat,depth,excluded,k,c,**evidence_options)
            valid=np.zeros(len(flat),bool);valid[supported]=True;valid=valid.reshape(-1,7).all(1)
            local=(triangles-c[:3,3])@c[:3,:3];q=local@k.T;uv=q[:,:,:2]/np.maximum(q[:,:,2:],1e-8)
            edges=uv[:,1:]-uv[:,:1];area=abs(edges[:,0,0]*edges[:,1,1]-edges[:,0,1]*edges[:,1,0])/2
            better=valid&(area>texture_score)
            texture_frame[better]=index;texture_score[better]=area[better];texture_sources[index]=(source,c,k.copy())

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
            source,c,k=texture_sources[int(index)]
            if predicted:rgb=source['color']
            else:
                if digest(source['source_path'])!=source['sha256']:raise ValueError('Texture source changed')
                rgb=cv2.cvtColor(cv2.imread(source['source_path']),cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2]
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
    result['points']=[[i,*p] for i,p in enumerate(cloud.vertices[::max(1,len(cloud.vertices)//20000)])]
    report={'source_scene_sha256':digest(args.scene),'support_frames':count,'minimum_agreeing_frames':3,'depth_tolerance_m':.04,
            'maximum_free_space_contradictions':'max(2, 0.15 * positive_support)', 'input_triangles':len(face_keep),'retained_triangles':len(mesh.faces),
            'input_points':len(cloud_keep),'retained_points':len(cloud.vertices),'elapsedSeconds':time.monotonic()-started,
            'texture_mapped_triangles':int((texture_frame>=0).sum()),'texture_source_frames':sorted(int(x) for x in np.unique(texture_frame) if x>=0),
            'mesh_sha256':digest(args.output/'static-scene.glb'),'cloud_sha256':digest(args.output/'dense-static.glb')}
    if predicted:
        report.pop('depth_tolerance_m')
        report.update(depth_relative_tolerance=.04,depth_source='LingBot estimated monocular z-depth',metric_validated=False)
    if args.mesh_from_points:
        report['reconstruction']={'method':'Open3D ball pivoting on validated observed points','radii_m':[spacing*r for r in (1.5,2,3)],'normal_radius_m':spacing*4,'normal_neighbors':30,'coordinate_change':False,'hidden_surfaces_completed':False}
        result['method'] += ' + measured-point surface reconstruction and source-photo texture'
    result['provenance']['static_consistency']=report
    result['limitations']=list(dict.fromkeys([*result['limitations'],'静态网格与点云只保留至少3个未遮挡深度帧支持的区域；未观测区域仍不补造。']))
    (args.output/'scene.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False));(args.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true')
    p.add_argument('--lingbot-run',type=Path,help='Native predicted-depth source; uses relative bounds instead of metric thresholds')
    p.add_argument('--mesh-from-points',action='store_true',help='Reconstruct the previously checked dense cloud before depth and texture verification')
    for name in ['scene','native-scene','analysis','output']:p.add_argument('--'+name,type=Path)
    args=p.parse_args();self_check() if args.self_check else run(args)
