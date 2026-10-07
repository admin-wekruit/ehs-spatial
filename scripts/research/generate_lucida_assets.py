"""Replay bounded RecGen calls from accepted, same-grid object evidence.

Run --self-check locally; --object-ids takes a comma-separated explicit object list.
Photos only go to the authorized private Modal function after its model-load check.
"""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def camera_depth(points, c2w, valid):
    """Row-vector world points -> native OpenCV camera z; invalid means unobserved."""
    points=np.asarray(points);c2w=np.asarray(c2w,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    if points.shape!=valid.shape+(3,) or c2w.shape!=(4,4) or not np.isfinite(c2w).all():
        raise ValueError('Invalid world pointmap / c2w / valid shapes')
    if not np.allclose(c2w[3],[0,0,0,1]) or not np.allclose(c2w[:3,:3].T@c2w[:3,:3],np.eye(3),atol=1e-4):
        raise ValueError('Expected rigid native camera-to-world transform')
    w2c=np.linalg.inv(c2w)
    z=points@w2c[2,:3]+w2c[2,3]
    keep=valid&np.isfinite(points).all(axis=-1)&(np.linalg.norm(points,axis=-1)>1e-6)&np.isfinite(z)&(z>0)
    return np.where(keep,z,0).astype(np.float32),keep


def source_grid_crop(source_rgb, mask, depth, K, source_to_canonical, original_mask=None):
    """Crop original pixels; sample canonical z and, for the old variant, mask."""
    A=np.asarray(source_to_canonical,dtype=np.float64)
    if A.shape!=(3,3) or not np.allclose(A[2],[0,0,1]) or not np.isfinite(A).all():
        raise ValueError('Expected recorded affine from source to canonical pixel centres')
    y,x=np.nonzero(mask)
    if not len(x):raise ValueError('Empty canonical object mask')
    if original_mask is not None:
        original_mask=np.asarray(original_mask)
        if original_mask.dtype!=bool or original_mask.shape!=(source_rgb.height,source_rgb.width) or not original_mask.any():
            raise ValueError('Expected a nonempty original-resolution boolean mask')
        y,x=np.nonzero(original_mask)
    corners=np.array([[x.min()-.5,y.min()-.5,1],[x.max()+.5,y.min()-.5,1],
                      [x.min()-.5,y.max()+.5,1],[x.max()+.5,y.max()+.5,1]])
    if original_mask is None:corners=corners@np.linalg.inv(A).T
    lower=corners[:,:2].min(axis=0);upper=corners[:,:2].max(axis=0)
    side=int(np.ceil((upper-lower).max()*1.4))  # 20% margin covers native 1.2x crop.
    left,top=np.floor((lower+upper-side)/2).astype(int)
    box=[int(left),int(top),int(left+side),int(top+side)]
    yy,xx=np.indices((side,side));u=xx+left;v=yy+top
    source_pixels=np.stack([u,v,np.ones_like(u)],axis=-1)
    canonical=source_pixels@A.T
    cx=np.floor(canonical[...,0]+.5).astype(int);cy=np.floor(canonical[...,1]+.5).astype(int)
    inside=(u>=0)&(v>=0)&(u<source_rgb.width)&(v<source_rgb.height)&(cx>=0)&(cy>=0)&(cx<mask.shape[1])&(cy<mask.shape[0])
    sampled_depth=np.zeros((side,side),np.float32);sampled_mask=np.zeros((side,side),np.uint8)
    sampled_depth[inside]=depth[cy[inside],cx[inside]]
    sampled_mask[inside]=(mask[cy[inside],cx[inside]] if original_mask is None
                          else original_mask[v[inside],u[inside]]).astype(np.uint8)*255
    crop=np.array([[1,0,-left],[0,1,-top],[0,0,1]],dtype=np.float64)
    K_crop=crop@np.linalg.inv(A)@K
    return {'rgb':np.asarray(source_rgb.crop(box).convert('RGB')),'depth':sampled_depth,
            'mask':sampled_mask,'camera_intrinsics':K_crop.astype(np.float32)}, {
        'variant':'robot-source-crop' if original_mask is None else 'original-rgb-crop',
        'source_crop_xyxy':box,'source_to_canonical_pixel_centres':A.tolist(),
        'crop_to_source_pixel_centres':np.linalg.inv(crop).tolist(),
        'K_crop':K_crop.tolist(),'K_derivation':'crop_translation @ inv(source_to_canonical) @ K_canonical',
        'RGB_resampling_before_native_preprocess':'none; original pixels, black only outside original bounds',
        'depth_and_mask_resampling':('nearest canonical pixel centre; no depth upsampling accuracy claim' if original_mask is None
                                    else 'depth: nearest canonical pixel centre; mask: accepted original pixels without resizing; no additional depth accuracy'),
        'mask_erosion_note':'same native 5x5 kernel now operates in source pixels, not canonical pixels'}


def payload_for_object(root, obj, source_crop=False):
    input_mode=obj.get('generation_input','canonical')
    if input_mode not in ['canonical','original_rgb_crop']:raise ValueError('Unknown evidence generation_input')
    original_crop=input_mode=='original_rgb_crop'
    if source_crop and original_crop:raise ValueError('Do not combine a new object input contract with the robot-only variant')
    manifest=json.loads((root/'manifest.json').read_text())
    sources={f['frame_id']:f for f in manifest['frames']}
    views=sorted(obj['views'],key=lambda v:(v['frame_id']!=obj['reference_frame'],v['frame_id']))
    if not views or views[0]['frame_id']!=obj['reference_frame']:
        raise ValueError('Reference frame is missing')
    arrays={'view_count':np.array(len(views),dtype=np.int32)};records=[]
    for i,view in enumerate(views):
        paths={key:root/view[key] for key in ['canonical_rgb_path','canonical_mask_path','pointmap_path',
                                            'content_valid_path','conf_path','K_path','c2w_path']}
        for p in paths.values():
            if not p.resolve().is_relative_to(root.resolve()) or not p.is_file():raise ValueError('Evidence path missing/outside run')
        rgb=np.asarray(Image.open(paths['canonical_rgb_path']).convert('RGB'))
        mask=np.load(paths['canonical_mask_path'],allow_pickle=False).astype(bool)
        points=np.load(paths['pointmap_path'],allow_pickle=False)
        valid=np.load(paths['content_valid_path'],allow_pickle=False).astype(bool)
        conf=np.load(paths['conf_path'],allow_pickle=False)
        K=np.load(paths['K_path'],allow_pickle=False).astype(np.float32)
        c2w=np.load(paths['c2w_path'],allow_pickle=False)
        if rgb.shape!=mask.shape+(3,) or valid.shape!=mask.shape or conf.shape!=mask.shape or K.shape!=(3,3):
            raise ValueError('Evidence grids do not match')
        depth,keep=camera_depth(points,c2w,valid&np.isfinite(conf)&(conf>=0.1))
        if not (mask&keep).any() or depth.max()>30:
            raise ValueError('No observed object depth or native depth would trigger upstream millimetre heuristic')
        model_inputs={'rgb':rgb,'mask':mask.astype(np.uint8)*255,'depth':depth,'camera_intrinsics':K}
        variant={}
        if source_crop or original_crop:
            original_path=root/view['rgb_path']
            if not original_path.resolve().is_relative_to(root.resolve()):raise ValueError('Original RGB outside run')
            if digest(original_path)!=sources[view['frame_id']]['sha256']:raise ValueError('Original RGB hash changed')
            original_rgb=Image.open(original_path).convert('RGB');original_mask=None
            frame=sources[view['frame_id']]
            if original_crop:
                mask_path=root/view['mask_path']
                if not mask_path.resolve().is_relative_to(root.resolve()):raise ValueError('Original mask outside run')
                if digest(mask_path)!=view['sha256']['mask.png'] or digest(paths['canonical_mask_path'])!=view['sha256']['canonical_mask.npy']:
                    raise ValueError('Accepted mask evidence hash changed')
                pixels=np.asarray(Image.open(mask_path))
                if pixels.shape!=(original_rgb.height,original_rgb.width) or not np.isin(pixels,[0,255]).all():
                    raise ValueError('Expected accepted original-resolution binary mask PNG')
                original_mask=pixels>0
                x0,y0,x1,y1=frame['content_rect_xyxy']
                if not all(isinstance(v,int) for v in [x0,y0,x1,y1]) or not (0<=x0<x1<=mask.shape[1] and 0<=y0<y1<=mask.shape[0]):
                    raise ValueError('Invalid declared canonical content rectangle')
                transform=frame.get('input_mask_transform')
                rh,rw=transform['resized_shape_hw'] if transform else (y1-y0,x1-x0)
                crop=transform['crop_xyxy'] if transform else [0,0,rw,rh]
                if (not all(isinstance(v,int) for v in [rh,rw,*crop]) or
                    not (0<=crop[0]<crop[2]<=rw and 0<=crop[1]<crop[3]<=rh) or
                    (crop[2]-crop[0],crop[3]-crop[1])!=(x1-x0,y1-y0)):
                    raise ValueError('Invalid source resize/crop operation')
                sx,sy=rw/original_rgb.width,rh/original_rgb.height
                expected_A=np.array([[sx,0,x0+(sx-1)/2-crop[0]],[0,sy,y0+(sy-1)/2-crop[1]],[0,0,1]])
                if not np.allclose(expected_A,frame['input_to_canonical_pixel_centres'],atol=1e-10,rtol=0):
                    raise ValueError('Original RGB dimensions disagree with recorded pixel affine')
                expected_mask=np.zeros(mask.shape,bool)
                expected_mask[y0:y1,x0:x1]=np.asarray(Image.fromarray(original_mask).resize((rw,rh),Image.Resampling.NEAREST).crop(crop))
                if not np.array_equal(mask,expected_mask):raise ValueError('Original and canonical accepted masks disagree')
                paths['original_mask_path']=mask_path
            model_inputs,variant=source_grid_crop(original_rgb,mask,depth,K,
                frame['input_to_canonical_pixel_centres'],original_mask)
            paths['original_rgb_path']=original_path
            if original_crop:
                from scipy.ndimage import binary_erosion
                eroded=binary_erosion(model_inputs['mask']>0,structure=np.ones((5,5),bool))
                supported=eroded&(model_inputs['depth']>0)
                if not supported.any():raise ValueError('Original mask has no observed depth after the official 5x5 erosion')
                variant['native_erosion_supported_pixels']=int(supported.sum())
        arrays.update({f'{i}_{key}':value for key,value in model_inputs.items()})
        records.append({'frame_id':view['frame_id'],'paths':{k:str(p.relative_to(root)) for k,p in paths.items()},
            'sha256':{k:digest(p) for k,p in paths.items()},'shape_hw':list(mask.shape),
            'object_mask_pixels':int(mask.sum()),'observed_object_depth_pixels':int((mask&keep).sum()),
            'observed_object_conf_quantiles':np.quantile(conf[mask&keep],[0,.1,.5,.9,1]).tolist(),
            'object_depth_quantiles':np.quantile(depth[mask&keep],[0,.05,.5,.95,1]).tolist(),
            'source_input_sha256':sources[view['frame_id']]['sha256'],
            'input_to_canonical_pixel_centres':sources[view['frame_id']]['input_to_canonical_pixel_centres'],
            'inference_shape_hw':list(model_inputs['mask'].shape),
            'inference_mask_pixels':int((model_inputs['mask']>0).sum()),
            'inference_observed_depth_pixels':int(((model_inputs['mask']>0)&(model_inputs['depth']>0)).sum()),
            **variant})
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    return stream.getvalue(),{'anchor_frame':views[0]['frame_id'],'source_views':records,
        'input_grid':('original RGB and accepted original-resolution mask crop; transformed K; nearest canonical depth' if original_crop else
                      'original RGB crop with explicitly transformed K and nearest sampled canonical depth/mask' if source_crop else
                      'canonical RGB, mask, depth and K without additional resampling'),
        'generation_input':'original_rgb_crop_with_canonical_mask' if source_crop else input_mode,
        'depth_derivation':'z of inv(native camera_to_world) @ native pointmap; content_valid, finite confidence>=0.1 and finite positive z; invalid=0',
        'depth_unit':f"native {(manifest.get('geometry') or {}).get('model') or 'source-geometry'} model-estimated units, not physically calibrated",
        'object_evidence_sha256':hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()}


def self_check():
    K=np.array([[10.,0,2],[0,20,1],[0,0,1]])
    yy,xx=np.indices((3,5));z=np.full((3,5),2.)
    camera=np.stack([(xx-2)*z/10,(yy-1)*z/20,z],axis=-1)
    c2w=np.array([[0.,0,1,3],[0,1,0,4],[-1,0,0,5],[0,0,0,1]])
    points=camera@c2w[:3,:3].T+c2w[:3,3]
    valid=np.ones(z.shape,bool);valid[1,2]=False;points[0,0]=np.nan
    conf=np.ones(z.shape);conf[0,1]=.05
    depth,keep=camera_depth(points,c2w,valid&np.isfinite(conf)&(conf>=0.1))
    assert np.allclose(depth[keep],2.) and depth[1,2]==0 and depth[0,0]==0 and depth[0,1]==0
    rays=np.stack([xx,yy,np.ones_like(xx)],axis=-1)@np.linalg.inv(K).T
    assert np.allclose((rays*depth[...,None])[keep],camera[keep])
    raw=np.array([[1.,2,3],[0,2,1],[-1,0,2]])
    cam2ncam=np.diag([2.,2,2,1]);cam2ncam[:3,3]=[1,3,-2]
    pose=np.eye(4);pose[:3,:3]*=.5;pose[:3,3]=[2,1,4]
    combined=np.linalg.inv(cam2ncam)@pose
    staged=(raw@pose[:3,:3].T+pose[:3,3]-cam2ncam[:3,3])/2
    assert np.allclose(raw@combined[:3,:3].T+combined[:3,3],staged)
    A=np.array([[.5,0,-.25],[0,.25,-.375],[0,0,1]])
    mask=np.zeros((3,5),bool);mask[1,1:4]=True
    source=Image.fromarray(np.arange(12*10*3,dtype=np.uint8).reshape(12,10,3))
    model,meta=source_grid_crop(source,mask,np.full((3,5),2,np.float32),K,A)
    left,top,right,bottom=meta['source_crop_xyxy']
    assert np.array_equal(model['rgb'],np.asarray(source.crop((left,top,right,bottom))))
    yy,xx=np.indices(model['depth'].shape)
    dest=np.stack([xx,yy,np.ones_like(xx)],axis=-1)
    canonical=(dest@np.asarray(meta['crop_to_source_pixel_centres']).T)@A.T
    assert np.allclose(dest@np.linalg.inv(model['camera_intrinsics']).T,canonical@np.linalg.inv(K).T,atol=1e-6)
    for iy,ix in np.argwhere(model['mask']>0):
        cy,cx=int(np.floor(canonical[iy,ix,1]+.5)),int(np.floor(canonical[iy,ix,0]+.5))
        assert mask[cy,cx] and model['depth'][iy,ix]==2
    original_mask=np.zeros((12,10),bool);original_mask[3:8,2:8]=True;original_mask[4,3]=False
    detailed,detail=source_grid_crop(source,mask,np.full((3,5),2,np.float32),K,A,original_mask)
    box=detail['source_crop_xyxy'];expected=np.asarray(Image.fromarray(original_mask).crop(box))
    assert np.array_equal(detailed['mask']>0,expected), 'Do not replace native mask detail with enlarged canonical pixels'
    assert detail['variant']=='original-rgb-crop' and np.array_equal(detailed['rgb'],np.asarray(source.crop(box)))
    try:source_grid_crop(source,mask,np.full((3,5),2,np.float32),K,A,original_mask[:-1])
    except ValueError:pass
    else:raise AssertionError('Wrong original mask dimensions accepted')
    stream=io.BytesIO();np.savez_compressed(stream,depth=depth);stream.seek(0)
    assert np.array_equal(np.load(stream,allow_pickle=False)['depth'],depth)
    print('self-check passed: rotated/translated c2w, camera-z, invalid pixels, K backprojection native pose composition, and original RGB/crop pixel-centre mapping')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/candidate-evaluation/lucida-replica-01')
    parser.add_argument('--object-ids',default='')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--prepare-inputs-only',action='store_true')
    parser.add_argument('--source-crop',action='store_true',help='One immutable original-RGB robot variant after baseline generation')
    args=parser.parse_args()
    if args.self_check:self_check();return
    root=args.root.resolve();generation=root/'generation';generation.mkdir(exist_ok=True)
    objects=json.loads((root/'evidence/objects.json').read_text())['objects']
    ids=args.object_ids.split(',') if args.object_ids else []
    if not ids or len(set(ids))!=len(ids):raise ValueError('Provide explicit unique --object-ids')
    if args.source_crop and ids!=['robot']:raise ValueError('Source-crop experiment is bounded to robot')
    selected=[]
    for oid in ids:
        if not oid or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in oid):raise ValueError('Unsafe object id')
        matches=[o for o in objects if o['object_id']==oid]
        if len(matches)!=1:raise ValueError('Unknown or duplicate evidence object: '+oid)
        selected.append(matches[0])
    if args.prepare_inputs_only:
        for obj in selected:
            payload,source=payload_for_object(root,obj,args.source_crop)
            print(obj['object_id'],len(payload),json.dumps(source))
        return
    envpath=generation/'environment-recgen/output.json'
    environment=json.loads(envpath.read_text())
    if environment['status']!='complete':raise ValueError('Private model environment check has not passed')
    sys.path.insert(0,str(REPO))
    from modal_apps.lucida_assets import app,generate_object,volume,read_outputs,CODE_REV,MODEL_REV
    deployed_app = os.environ.get('LUCIDA_DEPLOYED_APP')
    if deployed_app:
        import modal
        generate_object = modal.Function.from_name(deployed_app, 'generate_object')
    # On-prem (scripts/onprem/run_stage.py sets PANOPTES_ONPREM=1) nothing is billed by Modal: the GPU budget gate is off;
    # calls are still journaled in gpu-budget.json, marked onprem.
    onprem=os.environ.get('PANOPTES_ONPREM')=='1'
    # ponytail: one experiment process owns a simple file lock and sequential calls; no queue/service.
    with (generation/'gpu-budget.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledgerpath=generation/'gpu-budget.json'
        ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else {
            'limit_seconds':3600,'timeout_per_call_seconds':600,
            'accounting':'completed function wall seconds plus 30-second per-container overhead allowance; failed/interrupted calls charge full 600 seconds',
            'calls':[{'object_id':'environment-check','charged_seconds':environment['gpu_function_seconds']+30,
                      'gpu_function_seconds':environment['gpu_function_seconds'],'output_sha256':digest(envpath)}]}
        def saveledger():
            temp=ledgerpath.with_suffix('.tmp');temp.write_text(json.dumps(ledger,indent=2)+'\n');temp.replace(ledgerpath)
        with nullcontext() if deployed_app else app.run():
            for obj in selected:
                oid=obj['object_id'];out=generation/'variants/robot-source-crop' if args.source_crop else generation/oid
                if out.exists():raise FileExistsError('Preserve existing generation: '+str(out))
                if not onprem and sum(c['charged_seconds'] for c in ledger['calls'])+630>ledger['limit_seconds']:
                    raise RuntimeError('Insufficient remaining GPU budget for a bounded 600-second object call')
                payload,source=payload_for_object(root,obj,args.source_crop)
                if args.source_crop:
                    baseline=json.loads((generation/'robot/output.json').read_text())
                    if (baseline['seed']!=args.seed or baseline['status']!='complete' or
                        baseline['code_revision']!=CODE_REV or baseline['model_revision']!=MODEL_REV):
                        raise ValueError('Baseline seed/status/model pin mismatch')
                    byframe={v['frame_id']:v for v in source['source_views']}
                    if set(byframe)!={v['frame_id'] for v in baseline['source_views']}:raise ValueError('Variant view set changed')
                    for v in baseline['source_views']:
                        if any(byframe[v['frame_id']]['sha256'][k]!=sha for k,sha in v['sha256'].items()):
                            raise ValueError('Variant canonical evidence changed')
                    source['variant']='robot-source-crop'
                    source['canonical_baseline_output_sha256']=digest(generation/'robot/output.json')
                out.mkdir(parents=True);(out/'input.npz').write_bytes(payload)
                (out/'source.json').write_text(json.dumps(source,indent=2)+'\n')
                call={'object_id':oid,'output_dir':str(out.relative_to(generation)),
                      'variant':'robot-source-crop' if args.source_crop else None,
                      'generation_input':source['generation_input'],
                      'charged_seconds':630,'status':'reserved','seed':args.seed,**({'onprem':True} if onprem else {})}
                ledger['calls'].append(call);saveledger()
                begin=time.monotonic()
                try:
                    job_id=uuid.uuid4().hex
                    with volume.batch_upload() as upload:
                        upload.put_file(out/'input.npz',f'/jobs/{job_id}/input.npz')
                    call['private_volume_job_id']=job_id;saveledger()
                    remote_manifest=generate_object.remote(job_id,object_id=oid,seed=args.seed)
                    (out/'private-volume-manifest.json').write_text(json.dumps(remote_manifest,indent=2)+'\n')
                    result=read_outputs(remote_manifest)
                    record=json.loads(result['output.json'])
                    if record['source_payload_sha256']!=digest(out/'input.npz'):
                        raise ValueError('Remote inference input SHA mismatch')
                    record.update(source)
                    record['local_call_wall_seconds']=time.monotonic()-begin
                    record['source_json_sha256']=digest(out/'source.json')
                    result['output.json']=(json.dumps(record,indent=2)+'\n').encode()
                    for name,data in result.items():
                        if Path(name).name!=name:raise ValueError('Unexpected remote output filename')
                        (out/name).write_bytes(data)
                    call.update(status=record['status'],gpu_function_seconds=record['gpu_function_seconds'],
                        charged_seconds=min(630,max(record['gpu_function_seconds']+30,record['local_call_wall_seconds'])),output_sha256=digest(out/'output.json'))
                    saveledger();print(oid,record['status'],record.get('vertices'),record.get('faces'),flush=True)
                    if record['status']!='complete':raise RuntimeError(record.get('error','Remote inference failed'))
                except BaseException:
                    call['status']='failed_or_interrupted';saveledger();raise


if __name__=='__main__':main()
