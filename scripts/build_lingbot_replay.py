"""Export native LingBot RGB geometry into the existing replay contract.

Keeps its uncalibrated coordinate system separate from the RGB-D experiment.
SAM masks exclude segmented objects from static fusion; this does not solve dynamic tracking.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from build_replay_scene import media_spans, world_points
from reconstruct_room_rgb import digest, integrate, new_volume, pointmap_residuals, validate_frame


def source_transform(shape, target_width=518, patch=14):
    h,w=shape;rh=round(h*(target_width/w)/patch)*patch
    if rh<=0:raise ValueError('Invalid source aspect ratio')
    y0=max(0,(rh-target_width)//2)
    transform=np.array([[target_width/w,0,(target_width/w-1)/2],
        [0,rh/h,(rh/h-1)/2-y0],[0,0,1.]])
    return transform,rh,y0


def resize_mask(mask, shape):
    if mask.shape!=tuple(shape):raise ValueError('Mask is not in the source pixel domain')
    _,rh,y0=source_transform(shape)
    # Pixel centers follow the same resize/crop as the official RGB loader.
    return cv2.resize(mask.astype('uint8'),(518,rh),interpolation=cv2.INTER_NEAREST_EXACT)[y0:y0+min(rh,518)]>0


def read_prediction(path):
    with np.load(path,allow_pickle=False) as data:
        depth=data['depth'].squeeze(-1);confidence=data['depth_conf'];rgb=data['rgb'];k=data['k']
        # Official demo_render/data/loader.py treats saved extrinsic as W2C.
        # demo.postprocess's C2W comment is inconsistent with its consumers.
        w2c=np.eye(4);w2c[:3]=data['w2c'];c2w=np.linalg.inv(w2c)
    valid=validate_frame(rgb,depth,confidence,np.ones(depth.shape,bool),k,c2w)
    return depth,confidence,rgb,k,c2w,valid


def check_temporal_geometry(run, analysis):
    """Independent RGB tracks test camera direction; a same-frame roundtrip cannot."""
    execution=json.loads((run/'run.json').read_text());files=execution['frames']
    observed={f['sourceFrame']:f for f in analysis['frames']};shape=(analysis['height'],analysis['width'])
    records=[];lag=min(10,max(1,len(files)//4))
    for a,b in zip(files[::lag],files[lag::lag]):
        for entry in (a,b):
            if digest(run/entry['file'])!=entry['sha256']:raise ValueError('Temporal check input changed')
        da,ca,ra,ka,wa,va=read_prediction(run/a['file'])
        db,cb,rb,kb,wb,vb=read_prediction(run/b['file'])
        masks=[]
        for entry,conf in [(a,ca),(b,cb)]:
            mask=conf>=1.5
            for obj in observed.get(entry['sourceFrame'],{}).get('objects',[]):
                source=Path(analysis['_base'])/obj['maskUrl'];rgba=cv2.imread(str(source),-1)
                mask&=~resize_mask(rgba[:,:,3]>0,shape)
            masks.append(mask)
        ga=cv2.cvtColor(ra,cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(rb,cv2.COLOR_RGB2GRAY)
        p=cv2.goodFeaturesToTrack(ga,maxCorners=600,qualityLevel=.02,minDistance=6,mask=masks[0].astype('uint8')*255)
        if p is None:continue
        q,status,_=cv2.calcOpticalFlowPyrLK(ga,gb,p,None)
        back,back_status,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None)
        p=p[:,0];q=q[:,0];back=back[:,0];h,w=da.shape
        keep=status[:,0].astype(bool)&back_status[:,0].astype(bool)&(np.linalg.norm(back-p,axis=1)<1)
        keep&=(q[:,0]>=0)&(q[:,0]<w-1)&(q[:,1]>=0)&(q[:,1]<h-1)
        p=p[keep];q=q[keep]
        if not len(p):continue
        target=np.rint(q).astype(int);keep=masks[1][target[:,1],target[:,0]];p=p[keep];q=q[keep]
        if len(p)<20:continue
        src=np.rint(p).astype(int);z=da[src[:,1],src[:,0]]
        world=world_points(p,z,ka,wa);local=(world-wb[:3,3])@wb[:3,:3];uv=local@kb.T
        error=np.linalg.norm(uv[:,:2]/uv[:,2:]-q,axis=1)
        records.append({'source_frames':[a['sourceFrame'],b['sourceFrame']],'static_rgb_tracks':len(p),
            'median_reprojection_error_px':float(np.median(error)),'p95_reprojection_error_px':float(np.percentile(error,95))})
    passed=sum(r['median_reprojection_error_px']<=5 for r in records)
    report={'method':'background forward/backward RGB optical flow versus independently predicted depth/pose projection',
        'execution_sha256':digest(run/'run.json'),'pairs':records,'passing_pairs':passed,
        'threshold_median_pixels':5,'required_pair_fraction':.75,'groundtruth_used':False,
        'passed':len(records)>=3 and passed/len(records)>=.75}
    return report


def build(args):
    import open3d as o3d
    import trimesh
    started=time.monotonic();run=args.run
    plan=json.loads((run/'plan.json').read_text());execution=json.loads((run/'run.json').read_text())
    if execution['status']!='inference_complete' or execution['plan_sha256']!=digest(run/'plan.json'):
        raise ValueError('Require a complete source-bound native model execution')
    if execution['groundtruth_uploaded'] or execution['sensor_depth_uploaded']:
        raise ValueError('This comparison requires RGB-only inference')
    if digest(plan['source_video'])!=plan['source_video_sha256']:raise ValueError('Video changed')
    spans,source_shape=media_spans(plan['source_video'])
    analysis=json.loads(args.analysis.read_text())
    if analysis['provenance']['sourceVideoSha256']!=plan['source_video_sha256']:
        raise ValueError('Masks refer to a different video')
    temporal=check_temporal_geometry(run,{**analysis,'_base':str(args.analysis.parent)})
    if not temporal['passed']:
        raise ValueError('Cross-frame camera/depth does not follow static RGB tracks: '+json.dumps(temporal))
    observed={f['sourceFrame']:f for f in analysis['frames']}
    if len(execution['frames'])!=len(plan['frames']):raise ValueError('Incomplete prediction sequence')
    scene={'schema':'phase2-replay-scene-v1','coordinate_frame':'lingbot_native_monocular',
        'units':'uncalibrated_monocular','source_video_sha256':plan['source_video_sha256'],
        'method':'Official pretrained LingBot-Map streaming RGB + SAM object exclusion + Open3D TSDF',
        'points':[],'frames':[],'complete_room_accepted':False,
        'provenance':{'execution_sha256':digest(run/'run.json'),'plan_sha256':digest(run/'plan.json'),
            'analysis_sha256':digest(args.analysis),'sensor_depth':False,'gt_pose_input':False,
            'coordinate_alignment_applied':False,'camera_sampling':f"source frames every {plan['stride']} frames; held until next estimate"},
        'limitations':['纯RGB预测，原生尺度未标定，不能按米读取或直接叠加另一地图的人体模型。',
            '相机在抽帧时刻估计；播放间隔显示最近一次相机估计。',
            '静态融合排除已分割对象；已完成分析但检测为空的帧仍参与融合，缺少分析的帧不参与。漏检的移动物体仍可能留下残影。',
            '本样本用于独立几何对照，完整物体网格、人体形体及持久身份尚未接入。']}
    points=[];colors=[];diagnostics=[];volume=None;fused=0
    for i,(record,result) in enumerate(zip(plan['frames'],execution['frames'])):
        index=record['sourceFrame'];path=run/result['file']
        if result['sourceFrame']!=index or digest(path)!=result['sha256'] or abs(record['timeSec']-spans[index][0])>.001:
            raise ValueError('Prediction source index, hash or media timestamp differs')
        depth,confidence,rgb,k,c2w,valid=read_prediction(path)
        yy,xx=np.indices(depth.shape)
        world=world_points(np.column_stack((xx.ravel(),yy.ravel())),depth.ravel(),k,c2w).reshape(*depth.shape,3)
        h,w=depth.shape
        expected_h=round(source_shape[0]*(518/source_shape[1])/14)*14
        if (h,w)!=(min(expected_h,518),518):raise ValueError('Unexpected official crop domain')
        contract=pointmap_residuals(world,depth,k,c2w,valid)
        diagnostics.append({'sourceFrame':index,**contract})
        frame={'sourceFrame':index,'timeSec':record['timeSec'],
            'endTimeSec':plan['frames'][i+1]['timeSec'] if i+1<len(plan['frames']) else spans[-1][1],
            'c2w':c2w.tolist(),'objects':[]}
        scene['frames'].append(frame)
        if volume is None:
            reference=float(np.median(depth[valid]));voxel=reference*.005
            volume=new_volume(voxel)
            scene['provenance']['native_geometry']={'reference_depth_native':reference,'voxel_native':voxel,
                'confidence_threshold':1.5,'point_sampling_pixels':3,'metric_scale_validated':False}
        af=observed.get(index)
        if af is None:continue
        excluded=np.zeros((h,w),bool)
        for obj in af['objects']:
            path=(args.analysis.parent/obj['maskUrl']).resolve()
            if not path.is_relative_to(args.analysis.parent.resolve()):raise ValueError('Mask outside saved analysis')
            mask=cv2.imread(str(path),-1)
            if mask is None or mask.shape!=(*source_shape,4):raise ValueError('Invalid source mask raster')
            excluded|=resize_mask(mask[:,:,3]>0,source_shape)
        excluded=cv2.dilate(excluded.astype('uint8'),np.ones((5,5),'uint8'))>0
        static=valid&~excluded&(confidence>=1.5)
        filtered=np.where(static,depth,0).astype('float32')
        integrate(volume,rgb,filtered,k,c2w);fused+=1
        keep=static[::3,::3]
        points.append(world[::3,::3][keep]);colors.append(rgb[::3,::3][keep])
    if not points:raise ValueError('No static RGB geometry supported by this sequence')
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points)))
    pc.colors=o3d.utility.Vector3dVector(np.concatenate(colors)/255)
    cloud=pc.voxel_down_sample(voxel)
    while len(cloud.points)>300000:
        voxel*=1.2;cloud=pc.voxel_down_sample(voxel)
    mesh=volume.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):raise ValueError('TSDF produced no mesh')
    args.output.mkdir(parents=True,exist_ok=False)
    mesh_path=args.output/'static-scene.glb';cloud_path=args.output/'dense-static.glb'
    trimesh.Trimesh(vertices=np.asarray(mesh.vertices),faces=np.asarray(mesh.triangles),
        vertex_colors=np.rint(np.asarray(mesh.vertex_colors)*255).astype('uint8'),process=False).export(mesh_path)
    tcloud=trimesh.points.PointCloud(np.asarray(cloud.points),colors=np.rint(np.asarray(cloud.colors)*255).astype('uint8'))
    trimesh.Scene(tcloud).export(cloud_path);tcloud.export(args.output/'dense-static.ply')
    scene.update(meshUrl=mesh_path.name,pointCloudUrl=cloud_path.name,pointCloudCount=len(cloud.points))
    scene['points']=[[i,*xyz] for i,xyz in enumerate(np.asarray(cloud.points)[::max(1,len(cloud.points)//20000)])]
    report={'frames':len(scene['frames']),'fused_frames':fused,'points':len(cloud.points),'triangles':len(mesh.triangles),
        'mesh_sha256':digest(mesh_path),'cloud_sha256':digest(cloud_path),'cloud_voxel_native':voxel,
        'elapsed_seconds':time.monotonic()-started,'geometry_contract_per_frame':diagnostics,
        'points_unprojected_from_native_depth':True,'complete_room_accepted':False}
    scene['provenance']['export']=report
    scene['provenance']['temporal_geometry_check']=temporal
    for name,value in [('scene.json',scene),('metrics.json',report)]:
        (args.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='geometry_contract_per_frame'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','analysis','output']:p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())
