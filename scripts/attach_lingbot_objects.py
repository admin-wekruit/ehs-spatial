"""Reuse saved SAM/pose/body/photo-analysis evidence in a native LingBot map.

No provider calls, RGB-D geometry, GT trajectory or metric scale are imported.
Cached human predictions are refit in the new camera and checked per timestamp.
"""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from build_lingbot_replay import check_temporal_geometry, read_prediction, resize_mask, source_transform
from build_replay_scene import surface_joints, world_points
from build_video_body_models import camera_body
from build_video_object_models import observed_surface
from build_video_surfaces import export_surface
from reconstruct_room_rgb import digest


def read(path):return json.loads(Path(path).read_text())
def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')


@lru_cache(maxsize=8)
def body_prediction(folder):
    import trimesh
    source=read(folder/'input-manifest.json')
    for name in ['image','mask']:
        if digest(folder/(name+'.png'))!=source[name+'_sha256']:raise ValueError('Cached body source changed')
    result=read(folder/'provider-output.json')
    if len(result['metadata']['people'])!=1 or len(result['meshes'])!=1:raise ValueError('Expected one conditioned body')
    return trimesh.load(folder/'provider-mesh.ply',force='mesh',process=False),result['metadata']['people'][0]


def body_at(index,entity,cache,observations):
    """Only interpolate matching topology inside a continuously observed short track."""
    keys=cache.get(entity,{})
    if index in keys:
        mesh,person=body_prediction(keys[index]);return mesh.copy(),person,[index]
    a=max((i for i in keys if i<index),default=-1);b=min((i for i in keys if i>index),default=-1)
    if a<0 or b<0 or observations[b]['timeSec']-observations[a]['timeSec']>.38:return None
    if any(entity not in {o['entityId'] for o in observations[i]['objects']} for i in range(a,b+1)):return None
    ma,pa=body_prediction(keys[a]);mb,pb=body_prediction(keys[b])
    if not np.array_equal(ma.faces,mb.faces) or ma.vertices.shape!=mb.vertices.shape:raise ValueError('Body topology changed')
    t=(observations[index]['timeSec']-observations[a]['timeSec'])/(observations[b]['timeSec']-observations[a]['timeSec'])
    mesh=ma.copy();mesh.vertices=ma.vertices*(1-t)+mb.vertices*t
    person={key:(np.asarray(pa[key])*(1-t)+np.asarray(pb[key])*t).tolist()
        for key in ['keypoints_3d','keypoints_2d','pred_cam_t']}
    return mesh,person,[a,b]


def fit_body(mesh,person,k,transform,depth,mask,confidence,c2w):
    import open3d as o3d
    vertices,error=camera_body(mesh,person,k,transform)
    cast=o3d.t.geometry.RaycastingScene()
    cast.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices.astype('float32')),
        o3d.core.Tensor(mesh.faces.astype('uint32'))))
    h,w=mask.shape
    z=cast.cast_rays(cast.create_rays_pinhole(k,np.eye(4),w,h))['t_hit'].numpy()
    visible=np.isfinite(z)&(z>0)
    overlap=visible&mask&(confidence>=1.5)&np.isfinite(depth)&(depth>0)
    report={'depth_support_pixels':int(overlap.sum()),'silhouette_iou':float((visible&mask).sum()/max(1,(visible|mask).sum())),
        'pnp_error_px_p95':float(np.percentile(error,95)),'metric_scale_validated':False,
        'validation':'consistency with estimated monocular depth and source mask, not field accuracy'}
    if overlap.sum()<300:return None,{**report,'status':'insufficient_predicted_depth'}
    scale=float(np.median(depth[overlap]/z[overlap]));relative=abs(z[overlap]*scale-depth[overlap])/depth[overlap]
    report.update(scale_native_per_body_unit=scale,relative_depth_median=float(np.median(relative)),
        relative_depth_p95=float(np.percentile(relative,95)),
        quality_gate={'min_iou':.65,'max_relative_depth_median':.04,'max_relative_depth_p95':.10,'max_pnp_p95_px':5})
    accepted=report['silhouette_iou']>=.65 and report['pnp_error_px_p95']<=5 and np.median(relative)<=.04 and np.percentile(relative,95)<=.10
    report['status']='accepted_model_estimate' if accepted else 'rejected_alignment'
    if not accepted:return None,report
    mesh.vertices=vertices*scale@c2w[:3,:3].T+c2w[:3,3]
    mesh.visual.vertex_colors=[178,207,222,255]
    return mesh,report


def build(args):
    started=time.monotonic();scene=read(args.scene);analysis=read(args.analysis);plan=read(args.run/'plan.json');execution=read(args.run/'run.json')
    if (scene['coordinate_frame']!='lingbot_native_monocular' or scene['provenance']['execution_sha256']!=digest(args.run/'run.json')
        or scene['source_video_sha256']!=analysis['provenance']['sourceVideoSha256'] or scene['provenance']['analysis_sha256']!=digest(args.analysis)):
        raise ValueError('Map/mask/video source binding differs')
    temporal=check_temporal_geometry(args.run,{**analysis,'_base':str(args.analysis.parent)})
    if not temporal['passed']:raise ValueError('Camera/depth does not follow static RGB tracks')
    observations={f['sourceFrame']:f for f in analysis['frames']};shape=(analysis['height'],analysis['width'])
    transform,_,_=source_transform(shape)
    native={r['sourceFrame']:r for r in execution['frames']}
    cache={};cache_sources=[]
    for folder in sorted(args.body_run.glob('frame-*')):
        if not (folder/'provider-output.json').exists() or not (folder/'provider-mesh.ply').exists():continue
        s=read(folder/'input-manifest.json');idx=s['sourceFrame'];entity=s['entityId']
        if s['source_video_sha256']!=scene['source_video_sha256']:raise ValueError('Cached body belongs to a different video')
        obj=next(o for o in observations[idx]['objects'] if o['entityId']==entity)
        rgba=cv2.imread(str(args.analysis.parent/obj['maskUrl']),-1)
        if not np.array_equal(rgba[:,:,3]>0,cv2.imread(str(folder/'mask.png'),0)>0):raise ValueError('Cached body mask changed')
        cache.setdefault(entity,{})[idx]=folder
        cache_sources.append({'sourceFrame':idx,'entityId':entity,'path':str(folder.resolve()),
            'provider_output_sha256':digest(folder/'provider-output.json'),'provider_mesh_sha256':digest(folder/'provider-mesh.ply')})
    args.output.mkdir(parents=True,exist_ok=False)
    result=copy.deepcopy(scene)
    result['provenance']['temporal_geometry_check']=temporal
    for key in ['meshUrl','pointCloudUrl']:
        result[key]=os.path.relpath((args.scene.parent/result[key]).resolve(),args.output.resolve())
    result.update(bodyKeyframes=[],bodyInterpolation=[],staticObjects=[])
    body_audit=[];surface_audit=[];object_audit=[]
    for frame in result['frames']:
        index=frame['sourceFrame'];item=native[index];path=args.run/item['file']
        if digest(path)!=item['sha256']:raise ValueError('Native prediction changed')
        depth,confidence,rgb,k,c2w,valid=read_prediction(path)
        if not np.array_equal(c2w,np.asarray(frame['c2w'])):raise ValueError('Map camera changed')
        observed=observations[index];reference=float(np.median(depth[valid]));frame['objects']=[]
        frame['intrinsic']=k.tolist();frame['rasterSize']=[depth.shape[1],depth.shape[0]]
        for obj in observed['objects']:
            entity=obj['entityId'];source_mask=args.analysis.parent/obj['maskUrl']
            mask=resize_mask(cv2.imread(str(source_mask),-1)[:,:,3]>0,shape)
            support=mask&valid&(confidence>=1.5)
            keypoints=[]
            for joint in obj.get('keypoints',[]):
                keypoints.append(None if joint is None else [*(transform@np.array([*joint[:2],1]))[:2],joint[2]])
            joints=surface_joints(keypoints,np.where(valid,depth,0),support,k,c2w,reference*.06)
            item={'entityId':entity,'keypoints3d':joints,
                'bones':[b for b in obj.get('bones',[]) if joints[b[0]] is not None and joints[b[1]] is not None],
                'representation':'estimated_monocular_surface','world_motion':'insufficient_evidence'}
            if support.any():
                yy,xx=np.where(support);item['centroid']=np.median(world_points(np.column_stack((xx,yy)),depth[support],k,c2w),axis=0).tolist()
            frame['objects'].append(item)
            step=3;small=k.copy();small[:2]/=step
            v,faces,colors,pixels,_=observed_surface(rgb[::step,::step],depth[::step,::step],support[::step,::step],small,c2w,
                max_edge_m=reference*.025,depth_range=(0,np.inf))
            if len(faces):
                name=f'human-{index:05d}-{entity.rsplit("-",1)[-1]}.glb'
                export_surface(args.output/name,v,faces,colors)
                item['surface']={'meshUrl':name,'sourceFrame':index,'representation':'visible_monocular_surface',
                    'sha256':digest(args.output/name),'triangles':len(faces)}
                local=(v-c2w[:3,3])@c2w[:3,:3];uv=local@k.T
                error=float(np.max(abs(uv[:,:2]/uv[:,2:]-pixels*step)))
                if error>.001:raise ValueError('Human source pixels no longer align')
                surface_audit.append({'sourceFrame':index,'entityId':entity,'source_mask_sha256':digest(source_mask),
                    'prediction_sha256':native[index]['sha256'],'max_reprojection_error_px':error,**item['surface']})
            prediction=body_at(index,entity,cache,observations)
            if prediction is None:
                body_audit.append({'sourceFrame':index,'entityId':entity,'status':'no_supported_cached_body'});continue
            mesh,person,source_frames=prediction
            fitted,review=fit_body(mesh,person,k,transform,depth,mask,confidence,c2w)
            review.update(sourceFrame=index,entityId=entity,source_prediction_frames=source_frames)
            body_audit.append(review)
            if fitted is not None:
                name=f'body-{index:05d}-{entity.rsplit("-",1)[-1]}.glb';fitted.export(args.output/name)
                result['bodyKeyframes'].append({**review,'timeSec':frame['timeSec'],'meshUrl':name,
                    'representation':'inferred_anatomical_mesh','mesh_sha256':digest(args.output/name),
                    'topology_sha256':hashlib.sha256(np.asarray(fitted.faces,dtype=np.uint32).tobytes()).hexdigest(),'vertices':len(fitted.vertices)})
        if index%90==0:print(json.dumps({'sourceFrame':index,'surfaces':len(surface_audit),'bodies':len(result['bodyKeyframes'])}),flush=True)
    # Reuse source evidence and language analysis; rebuild only XYZ in the new map.
    original=read(args.reuse_scene)
    if original['source_video_sha256']!=scene['source_video_sha256']:raise ValueError('Object evidence from another video')
    for obj in original.get('staticObjects',[]):
        source=obj['source'];idx=source['sourceFrame']
        if idx not in native:raise ValueError('Object source frame was not reconstructed; use a denser input plan')
        frame=next(f for f in result['frames'] if f['sourceFrame']==idx)
        depth,confidence,rgb,k,c2w,valid=read_prediction(args.run/native[idx]['file'])
        mask_path=(args.reuse_scene.parent/source['maskUrl']).resolve();rgba=cv2.imread(str(mask_path),-1)
        mask=resize_mask(rgba[:,:,3]>0,shape)&valid&(confidence>=1.5)
        v,faces,colors,_,_=observed_surface(rgb,depth,mask,k,c2w,max_edge_m=float(np.median(depth[valid]))*.025,depth_range=(0,np.inf))
        if not len(faces):
            object_audit.append({'entityId':obj['entityId'],'status':'no_supported_predicted_surface'});continue
        name=obj['entityId']+'.glb';export_surface(args.output/name,v,faces,colors)
        item=copy.deepcopy(obj);item['meshUrl']=name
        # A held 3D sample can span several video frames. Source masks belong
        # only to the original video frame, never to the entire hold interval.
        observed=observations[idx]
        if source['timeSec']!=observed['timeSec'] or source['endTimeSec']!=observed['endTimeSec']:
            raise ValueError('Object mask source interval differs from original video frame')
        for key in ['maskUrl','imageUrl']:item['source'][key]=os.path.relpath((args.reuse_scene.parent/source[key]).resolve(),args.output.resolve())
        report={'entityId':obj['entityId'],'sourceFrame':idx,'status':'estimated_monocular_surface',
            'original_observation_sha256':digest(args.reuse_scene.parent/obj['provenanceUrl']),
            'source_mask_sha256':digest(mask_path),'source_image_sha256':digest(args.reuse_scene.parent/source['imageUrl']),
            'native_prediction_sha256':native[idx]['sha256'],'mesh_sha256':digest(args.output/name),
            'semantic_review_reused':bool(obj.get('semanticReview')),'metric_scale_validated':False,'hidden_surface_completed':False}
        item['provenanceUrl']=obj['entityId']+'.json';save(args.output/item['provenanceUrl'],report)
        result['staticObjects'].append(item);object_audit.append(report)
    result['humanSurfaces']={'representation':'visible_monocular_surface','samplingPixels':3,'count':len(surface_audit),
        'coordinateFrame':scene['coordinate_frame'],'hidden_body_completed':False}
    result['method']+=' + reused SAM3/RTMPose evidence + cached SAM3D Body refit + object photo analysis'
    result['limitations']=[s for s in result['limitations'] if '尚未接入' not in s]
    result['limitations']+=[f"每 {plan['stride']} 帧重建一次并按源时间回放；人体形状为预训练模型估计，短间隔姿态可来自缓存插值，逐帧重新检查掩码和预测深度。",
        '单目深度与人体的一致性检查不等于现场几何已验证；此版本未输出米制速度。',
        f"{len(result['staticObjects'])} 个家具/设备分割复用照片语义，表面位置由 LingBot 重新计算；跨时间身份和不可见背面尚未完成。"]
    result['provenance']['reused_evidence']={'body_cache':str(args.body_run.resolve()),'cached_predictions':len(cache_sources),
        'original_object_scene_sha256':digest(args.reuse_scene),'new_provider_calls':0,'sensor_depth_used':False,
        'cache_source_records_sha256':hashlib.sha256(json.dumps(cache_sources,sort_keys=True).encode()).hexdigest()}
    summary={'elapsed_seconds':time.monotonic()-started,'surfaces':len(surface_audit),'body_observations':len(body_audit),
        'accepted_bodies':len(result['bodyKeyframes']),'objects':len(result['staticObjects']),'new_provider_calls':0}
    save(args.output/'evidence.json',{'summary':summary,'cache':cache_sources,'bodies':body_audit,'surfaces':surface_audit,'objects':object_audit})
    result['provenance']['evidence_sha256']=digest(args.output/'evidence.json')
    save(args.output/'scene.json',result);save(args.output/'metrics.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','scene','analysis','body-run','reuse-scene','output']:p.add_argument('--'+name,type=Path,required=True)
    build(p.parse_args())
