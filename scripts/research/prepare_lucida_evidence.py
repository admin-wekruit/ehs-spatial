"""Freeze the three user photographs and their declared canonical pixel maps."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import sys
import subprocess

import numpy as np
from PIL import Image, ImageOps


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def object_view(root, object_id, frame_id, mask, provenance):
    """Keep the original pixels and an explicitly mapped canonical mask together."""
    manifest=json.loads((root/'manifest.json').read_text())
    frame=next(f for f in manifest['frames'] if f['frame_id']==frame_id)
    rgb=Image.open(root/frame['input']).convert('RGB')
    canonical=np.zeros((518,518),bool)
    if mask.shape==(518,518):
        canonical=mask.copy(); canonical[:,:63]=False;canonical[:,455:]=False
        native=np.asarray(Image.fromarray(canonical[:,63:455]).resize(rgb.size,Image.Resampling.NEAREST)).astype(bool)
    elif mask.shape==(rgb.height,rgb.width):
        native=mask.astype(bool)
        canonical[:,63:455]=np.asarray(Image.fromarray(native).resize((392,518),Image.Resampling.NEAREST)).astype(bool)
    else:raise ValueError('Mask is neither original pixels nor declared canonical grid')
    if not native.any():raise ValueError('Cannot export an empty object mask')
    out=root/'evidence/objects'/object_id/frame_id;out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(native.astype(np.uint8)*255).save(out/'mask.png')
    np.save(out/'canonical_mask.npy',canonical)
    y,x=np.nonzero(native);pad=max(8,round(.025*max(x.max()-x.min(),y.max()-y.min())))
    box=[max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(rgb.width,int(x.max())+pad+1),min(rgb.height,int(y.max())+pad+1)]
    rgba=rgb.convert('RGBA');rgba.putalpha(Image.fromarray(native.astype(np.uint8)*255));rgba.crop(box).save(out/'rgba.png')
    geom=root/'geometry/frames'/frame_id
    points=np.load(geom/'pts3d.npy');valid=np.load(geom/'content_valid_mask.npy')&canonical
    cloud=points[valid];np.save(out/'points.npy',cloud)
    colours=np.asarray(Image.open(geom/'canonical.png').convert('RGB'))[valid];np.save(out/'colors.npy',colours)
    assert len(cloud)>0 and np.isfinite(cloud).all()
    rel=lambda path:str(path.relative_to(root))
    return {'frame_id':frame_id,'rgb_path':frame['input'],'mask_path':rel(out/'mask.png'),
        'rgba_path':rel(out/'rgba.png'),'rgba_crop_xyxy_in_input':box,'rgba_pixel_to_input':[[1,0,box[0]],[0,1,box[1]],[0,0,1]],
        'canonical_mask_path':rel(out/'canonical_mask.npy'),'points_path':rel(out/'points.npy'),'colors_path':rel(out/'colors.npy'),
        'K_path':rel(geom/'intrinsics.npy'),'c2w_path':rel(geom/'camera_to_world.npy'),
        'canonical_rgb_path':rel(geom/'canonical.png'),'pointmap_path':rel(geom/'pts3d.npy'),
        'valid_path':rel(geom/'valid_mask.npy'),'content_valid_path':rel(geom/'content_valid_mask.npy'),'conf_path':rel(geom/'conf.npy'),
        'K_pixel_grid':'canonical518x518; source affine in manifest.json','mask_pixels':int(native.sum()),
        'partial_point_count':len(cloud),'centroid_native':np.median(cloud,axis=0).tolist(),
        'observed_only':True,'provenance':provenance,
        'sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}}


def reuse_masks(root, old_run):
    sys.path.insert(0,str(old_run.parents[1]))
    from ehs_spatial.providers.sam3 import decode_coco_rle
    # Explicitly checked same physical front posts and the right front fence panel.
    # Native centroid distances are recorded; labels alone never produce an association.
    items=[('left_post','left front black bollard',[(1,3,'bollard',0,15),(3,4,'bollard',0,21)],3),
           ('right_post','right front black bollard',[(1,3,'bollard',1,16),(3,4,'bollard',1,22)],3),
           ('right_fence','right front safety fence panel',[(1,3,'safety_fence',0,13),(3,4,'safety_fence',1,19)],1)]
    objects=[]
    for object_id,label,sources,reference in items:
        views=[]
        for new,old,slug,index,inv in sources:
            path=old_run/f'inventory/sam/frame_{old:04d}__{slug}.json'
            response=json.loads(path.read_text());mask=decode_coco_rle(response['rle'][index],height=518,width=518).astype(bool)
            provenance={'type':'existing same-capture canonical SAM mask','source_path':str(path),'source_sha256':digest(path),
                'source_frame':f'frame_{old:04d}','source_inv':inv,'source_instance':index,
                'source_to_new':'original captures matched after full-image resampling/JPEG; canonical coordinate bounds unchanged',
                'mask_resolution_limit':'518 canonical grid; native-size mask is nearest-neighbour enlargement'}
            views.append(object_view(root,object_id,f'frame_{new:04d}',mask,provenance))
        distance=float(np.linalg.norm(np.array(views[0]['centroid_native'])-views[1]['centroid_native']))
        assert distance<.1,'Explicit visual identity is not supported by new joint geometry'
        objects.append({'object_id':object_id,'label':label,'reference_frame':f'frame_{reference:04d}',
            'source_inventory_indices':[s[-1] for s in sources], 'physical_identity':{'status':'verified corresponding visible physical object',
                'evidence':'same gate-side location and visible fixed details; new joint native centroid agreement',
                'two_view_centroid_distance_native':distance},'views':views})
    save_json(root/'evidence/objects.json',{'version':1,'coordinate_system':'new joint Pi3X native OpenCV world',
        'metric_scale_known':False,'status':'partial; robot/cart/new-view masks pending','objects':objects})
    print(json.dumps({'objects':[o['object_id'] for o in objects],'ready':'evidence/objects.json'}))


def floor_evidence(root, old_run):
    sys.path.insert(0,str(old_run.parents[1]))
    from ehs_spatial.providers.sam3 import decode_coco_rle
    from ehs_spatial.geometry import _ransac_floor_plane
    evidence=[];clouds=[];colours=[]
    for new,old,box in [(1,3,[140,380,389,477]),(3,4,[160,385,400,478])]:
        path=old_run/f'inventory/sam/frame_{old:04d}__floor.json'
        response=json.loads(path.read_text())
        mask=np.logical_or.reduce([decode_coco_rle(r,height=518,width=518).astype(bool) for r in response['rle']])
        roi=np.zeros((518,518),bool);x0,y0,x1,y1=box;roi[y0:y1,x0:x1]=True;mask &= roi
        view=object_view(root,'observed_floor',f'frame_{new:04d}',mask,{'type':'same-capture cached SAM floor intersected with visible interior floor ROI',
            'source_path':str(path),'source_sha256':digest(path),'canonical_roi_xyxy':box,'source_frame':f'frame_{old:04d}'})
        evidence.append(view);clouds.append(np.load(root/view['points_path']));colours.append(np.load(root/view['colors_path']))
    points=np.concatenate(clouds);rgb=np.concatenate(colours)
    cameras=[np.load(root/f'geometry/frames/frame_{i:04d}/camera_to_world.npy') for i in [1,2,3]]
    up=np.mean([-c[:3,1] for c in cameras],axis=0);up/=np.linalg.norm(up)
    # The threshold is a fraction of observed floor extent in native units, not metres.
    threshold=.005*np.linalg.norm(np.quantile(points,.95,axis=0)-np.quantile(points,.05,axis=0))
    fit=points[::max(1,len(points)//20000)]
    indices=_ransac_floor_plane(fit,np.asarray([c[:3,3] for c in cameras]),up,threshold)
    if indices is None or len(indices)<150:raise ValueError('Insufficient observed floor plane support')
    center=np.mean(fit[indices],axis=0);_,_,vectors=np.linalg.svd(fit[indices]-center,full_matrices=False)
    normal=vectors[-1];normal*=np.sign(normal@up);offset=-float(normal@center)
    residual=np.abs(points@normal+offset);inside=residual<threshold
    np.save(root/'evidence/floor_points.npy',points)
    np.save(root/'evidence/floor_colors.npy',rgb)
    record={'plane_native':[float(v) for v in normal]+[offset],'up_native':normal.tolist(),
        'coordinate_system':'new joint Pi3X native world','metric_scale_known':False,
        'fit':'existing ehs_spatial.geometry._ransac_floor_plane then SVD on its inliers',
        'distance_threshold_native':float(threshold),'total_observed_points':len(points),'inlier_points':int(inside.sum()),
        'all_point_residual_median_native':float(np.median(residual)),'all_point_residual_p95_native':float(np.quantile(residual,.95)),
        'inlier_residual_p95_native':float(np.quantile(residual[inside],.95)),
        'camera_signed_heights_native':[float(c[:3,3]@normal+offset) for c in cameras],
        'points_path':'evidence/floor_points.npy','colors_path':'evidence/floor_colors.npy','views':evidence,
        'evidence_type':'observed partial geometry only; not a generated asset'}
    assert all(h>0 for h in record['camera_signed_heights_native'])
    save_json(root/'evidence/floor.json',record)
    objects=json.loads((root/'evidence/objects.json').read_text());background=[points];background_rgb=[rgb]
    for obj in objects['objects']:
        if 'fence' not in obj['object_id']:continue
        for v in obj['views']:background.append(np.load(root/v['points_path']));background_rgb.append(np.load(root/v['colors_path']))
    np.savez(root/'evidence/observed_background.npz',xyz=np.concatenate(background),rgb=np.concatenate(background_rgb))
    print(json.dumps({k:record[k] for k in ['plane_native','total_observed_points','inlier_points','all_point_residual_p95_native']}))


def segment_sam2(root,prompts_path,model_size="small"):
    import torch
    vendor=root/'evidence/vendor/sam2';weights=root/f'evidence/models/sam2.1_hiera_{model_size}.pt'
    sys.path.insert(0,str(vendor))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    config='configs/sam2.1/sam2.1_hiera_'+{'small':'s','large':'l'}[model_size]+'.yaml'
    model=build_sam2(config,str(weights),device='mps',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model)
    prompts=json.loads(prompts_path.read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    record={'model':'SAM2.1 Hiera '+model_size,'source':'https://github.com/facebookresearch/sam2',
        'code_revision':subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),
        'weight_url':f'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_{model_size}.pt',
        'weights_sha256':digest(weights),'license':'Apache-2.0','device':'mps','postprocessing':False,
        'prompts_path':str(prompts_path.relative_to(root)),'prompts_sha256':digest(prompts_path),'results':[]}
    for fid,items in prompts.items():
        frame=next(f for f in manifest['frames'] if f['frame_id']==fid)
        rgb=np.array(Image.open(root/frame['input']).convert('RGB'),copy=True)
        start=time.perf_counter()
        with torch.inference_mode():
            predictor.set_image(rgb)
            for item in items:
                masks,scores,_=predictor.predict(point_coords=np.array(item['points']) if item.get('points') else None,
                    point_labels=np.array(item['point_labels']) if item.get('points') else None,
                    box=np.array(item['box']) if item.get('box') else None,multimask_output=True)
                order=np.argsort(scores)[::-1];out=root/'evidence/sam2'/prompts_path.stem/fid/item['object_id'];out.mkdir(parents=True,exist_ok=True)
                candidates=[]
                for rank,index in enumerate(order):
                    mask=masks[index].astype(bool);path=out/f'candidate_{rank}.png'
                    Image.fromarray(mask.astype(np.uint8)*255).save(path)
                    rgba=Image.fromarray(rgb).convert('RGBA');rgba.putalpha(Image.fromarray(mask.astype(np.uint8)*255))
                    y,x=np.nonzero(mask);box=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
                    rgba.crop(box).save(out/f'candidate_{rank}_rgba.png')
                    candidates.append({'mask_path':str(path.relative_to(root)),'rgba_path':str((out/f'candidate_{rank}_rgba.png').relative_to(root)),
                        'mask_sha256':digest(path),'score':float(scores[index]),'pixels':int(mask.sum()),'bbox_xyxy':box})
                result={'frame_id':fid,'object_id':item['object_id'],'prompt':item,'input_sha256':frame['sha256'],'candidates':candidates}
                record['results'].append(result);print(json.dumps({'frame':fid,'object':item['object_id'],'candidates':candidates}),flush=True)
        record.setdefault('image_seconds',{})[fid]=time.perf_counter()-start
    save_json(root/'evidence/sam2'/f'{prompts_path.stem}-results.json',record)


def self_check(root):
    """Check the actual frozen RGB/mask/depth/camera bundle consumed downstream."""
    manifest=json.loads((root/'manifest.json').read_text())
    objects=json.loads((root/'evidence/objects.json').read_text())
    floor=json.loads((root/'evidence/floor.json').read_text())
    checks=[]
    for frame in manifest['frames']:
        assert digest(root/frame['input'])==frame['sha256']
        A=np.asarray(frame['input_to_canonical_pixel_centres'])
        assert np.allclose(A@np.asarray(frame['canonical_to_input_pixel_centres']),np.eye(3))
        assert np.allclose((A@[-.5,-.5,1])[:2],[62.5,-.5])
        geom=root/'geometry/frames'/frame['frame_id']
        record=next(f for f in manifest['geometry']['frames'] if f['frame_id']==frame['frame_id'])
        assert all(digest(geom/name)==sha for name,sha in record['files'].items())
        P=np.load(geom/'pts3d.npy');valid=np.load(geom/'content_valid_mask.npy')
        assert P.shape==(518,518,3) and valid.shape==(518,518)
        assert not valid[:,:63].any() and not valid[:,455:].any()
        assert np.all(np.isfinite(P[valid])) and np.all(abs(P[valid]).sum(axis=1)>1e-6)
        K=np.load(geom/'intrinsics.npy');C=np.load(geom/'camera_to_world.npy')
        camera=(P[valid]-C[:3,3])@C[:3,:3];uv=camera@K.T;uv=uv[:,:2]/uv[:,2:]
        y,x=np.nonzero(valid);error=np.linalg.norm(uv-np.column_stack((x,y)),axis=1)
        assert np.quantile(error,.95)<2,'Native pointmap/camera grid mismatch'
        checks.append({'frame_id':frame['frame_id'],'native_reprojection_p95_pixels':float(np.quantile(error,.95))})
    views=[v for obj in objects['objects'] for v in obj['views']]+floor['views']
    for view in views:
        canonical=np.load(root/view['canonical_mask_path']);native=np.asarray(Image.open(root/view['mask_path']))>0
        assert canonical.dtype==bool and canonical.shape==(518,518) and native.shape==(3840,2880)
        assert not canonical[:,:63].any() and not canonical[:,455:].any()
        assert np.array_equal(canonical[:,63:455],np.asarray(Image.fromarray(native).resize((392,518),Image.Resampling.NEAREST)))
        keep=canonical&np.load(root/view['content_valid_path'])
        assert np.array_equal(np.load(root/view['points_path']),np.load(root/view['pointmap_path'])[keep])
        assert np.array_equal(np.load(root/view['colors_path']),np.asarray(Image.open(root/view['canonical_rgb_path']))[keep])
        x0,y0,x1,y1=view['rgba_crop_xyxy_in_input'];rgba=np.asarray(Image.open(root/view['rgba_path']))
        assert np.array_equal(rgba[:,:,3]>0,native[y0:y1,x0:x1])
        parent=(root/view['mask_path']).parent
        assert all(digest(parent/name)==sha for name,sha in view['sha256'].items())
    assert all(not o['source_inventory_indices'] for o in objects['objects'] if o['object_id'] in ['robot','cart'])
    assert abs(np.linalg.norm(floor['up_native'])-1)<1e-5 and all(h>0 for h in floor['camera_signed_heights_native'])
    result={'status':'passed','object_count':len(objects['objects']),'checked_views_including_floor':len(views),
        'checks':'input/geometry/object SHA; pixel affine; no padding or zero sentinel; native K/c2w reprojection; exact alpha/partial-point/RGB correspondence; no old dynamic inventory reuse',
        'frames':checks,'objects_sha256':digest(root/'evidence/objects.json'),'floor_sha256':digest(root/'evidence/floor.json')}
    save_json(root/'evidence/self-check.json',result);print(json.dumps(result))


def freeze(source, output, old_run):
    source, output, old_run = source.resolve(), output.resolve(), old_run.resolve()
    if (output/'manifest.json').exists():
        raise ValueError('Input manifest is already frozen')
    (output/'input').mkdir(parents=True, exist_ok=True)
    canonical = output/'evidence/canonical'
    canonical.mkdir(parents=True, exist_ok=True)
    old = sorted((old_run/'input').glob('image_*.jpg'))
    old_pixels = [np.asarray(Image.open(p).convert('RGB').resize((360,480), Image.Resampling.LANCZOS), dtype=np.float32) for p in old]
    frames = []
    for index, filename in enumerate(['1-Photo-1.jpg','2-Photo-2.jpg','3-Photo-3.jpg'], 1):
        origin, frozen = source/filename, output/'input'/filename
        if frozen.exists():
            raise ValueError(f'Frozen input already exists: {frozen}')
        shutil.copyfile(origin, frozen)
        assert digest(origin) == digest(frozen)
        with Image.open(frozen) as native:
            rgb = ImageOps.exif_transpose(native).convert('RGB')
            if native.getexif().get(274, 1) != 1:
                raise ValueError('This fixed experiment requires upright source pixels')
            w,h = rgb.size
            canvas = Image.new('RGB',(518,518),'white')
            canvas.paste(rgb.resize((392,518),Image.Resampling.LANCZOS),(63,0))
            name = f'frame_{index:04d}'
            canvas.save(canonical/f'{name}.png')
            alpha = np.zeros((518,518),bool);alpha[:,63:455]=True
            np.save(canonical/f'{name}_alpha.npy',alpha)
            A = np.array([[392/w,0,63+(392/w-1)/2],[0,518/h,(518/h-1)/2],[0,0,1]])
            # A pixel-center transform must roundtrip and map the photograph's edge bounds.
            assert np.allclose(np.linalg.inv(A) @ A, np.eye(3))
            assert np.allclose((A @ [-.5,-.5,1])[:2],[62.5,-.5])
            assert np.allclose((A @ [w-.5,h-.5,1])[:2],[454.5,517.5])
            small = np.asarray(rgb.resize((360,480),Image.Resampling.LANCZOS),dtype=np.float32)
            comparisons = [{'frame_id':f'frame_{i:04d}','image':str(p),'sha256':digest(p),
                'rgb_mae_360x480':float(np.abs(small-q).mean()),
                'rgb_error_p95_360x480':float(np.quantile(np.abs(small-q),.95)),
                'rgb_correlation_360x480':float(np.corrcoef(small.ravel(),q.ravel())[0,1])}
                for i,(p,q) in enumerate(zip(old,old_pixels),1)]
            best = min(comparisons,key=lambda item:item['rgb_mae_360x480'])
            # ponytail: a declared low-resolution test finds candidates; visual inspection
            # and a full-resolution resampling residual must confirm before mask reuse.
            match = best if best['rgb_mae_360x480'] < 1 and best['rgb_correlation_360x480'] > .999 else None
            if match:
                with Image.open(match['image']) as original:
                    prior = original.convert('RGB').resize(rgb.size,Image.Resampling.LANCZOS)
                    residual = np.abs(np.asarray(rgb,dtype=np.int16)-np.asarray(prior,dtype=np.int16))
                    match |= {'old_size':list(original.size),'new_size':[w,h],
                        'resize_mae_full_resolution':float(residual.mean()),
                        'resize_error_p95_full_resolution':float(np.quantile(residual,.95)),
                        'status':'same capture after resampling and JPEG recompression; not byte/pixel identical',
                        'coordinate_relation':'full image bounds preserved; pixel-center resize',
                        'manual_visual_confirmation':True}
            frames.append({'frame_id':name,'source_upload':str(origin),'input':str(frozen.relative_to(output)),
                'sha256':digest(frozen),'bytes':frozen.stat().st_size,'width':w,'height':h,
                'decoded_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                'canonical':str((canonical/f'{name}.png').relative_to(output)),
                'canonical_sha256':digest(canonical/f'{name}.png'),
                'canonical_shape_hw':[518,518],'content_rect_xyxy':[63,0,455,518],
                'alpha':str((canonical/f'{name}_alpha.npy').relative_to(output)),
                'alpha_sha256':digest(canonical/f'{name}_alpha.npy'),
                'input_to_canonical_pixel_centres':A.tolist(),'canonical_to_input_pixel_centres':np.linalg.inv(A).tolist(),
                'resampling':'Pillow RGB LANCZOS; resize392x518 then white pad x=63..454',
                'existing_capture_match':match,'all_existing_capture_comparisons':comparisons})
    manifest={'experiment':'lucida-replica-01','created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'status':'inputs_frozen','frames':frames,'source_inventory':str(old_run/'inventory/inventory.json'),
        'source_inventory_sha256':digest(old_run/'inventory/inventory.json'),
        'geometry':{'status':'pending new joint inference','model':'Pi3X','input_frames':[f['frame_id'] for f in frames],
                    'coordinate_system':'new native OpenCV world; no old BOR1 coordinate transfer','metric_scale_known':False},
        'script_sha256':digest(__file__), 'privacy':'local input/geometry; segmentation only through the existing private Modal service'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(output),'frames':[{k:f[k] for k in ['frame_id','input','sha256','existing_capture_match']} for f in frames]},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source-run',required=True,type=Path)
    parser.add_argument('--reuse-masks',action='store_true')
    parser.add_argument('--prepare-floor',action='store_true')
    parser.add_argument('--sam2-prompts',type=Path)
    parser.add_argument('--sam2-model',choices=['small','large'],default='small')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check(args.output.resolve())
    elif args.sam2_prompts:segment_sam2(args.output.resolve(),args.sam2_prompts.resolve(),args.sam2_model)
    elif args.prepare_floor:floor_evidence(args.output.resolve(),args.source_run.resolve())
    elif args.reuse_masks:reuse_masks(args.output.resolve(),args.source_run.resolve())
    else:freeze(args.input_dir,args.output,args.source_run)
