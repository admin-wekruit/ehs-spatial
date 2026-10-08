"""Freeze object views in the existing capture contract."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

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
