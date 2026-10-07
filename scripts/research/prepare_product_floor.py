"""Extract observed support for the product's saved native geometric floor.

The product already fits this plane from depth, independently of semantic labels.
This step retains only actual supported pixels; it never creates a planar patch.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from generate_lucida_assets import digest, camera_depth


def prepare(root, ehs_repo):
    root = Path(root).resolve()
    sys.path.insert(0, str(ehs_repo))
    from ehs_spatial.geometry import _FLOOR_NORMAL_MAX_ANGLE_COS
    manifest = json.loads((root/'manifest.json').read_text())
    source = manifest.get('source_floor') or {}
    hashes = manifest['source_files_sha256']
    relative = source.get('path')
    if relative != 'scene.json' or not source.get('sha256') or hashes.get(relative) != source['sha256']:
        raise ValueError('Missing SHA-bound source scene.json native floor evidence')

    def verified(relative):
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != hashes.get(relative):
            raise ValueError('Missing or changed source floor input: '+relative)
        return path

    scene = json.loads(verified(relative).read_text())
    plane = np.asarray(scene.get('floor_plane'), float)
    scale = scene.get('scale_factor')
    if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-9:
        raise ValueError('Source scene has no finite native fitted floor plane')
    if not isinstance(scale, (int,float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Source floor fit has no positive scale_factor for its support threshold')
    plane /= np.linalg.norm(plane[:3])
    # Same final residual gate used by geometry._fit_floor; no new metric anchor.
    threshold = .03/scale
    views, clouds, cameras = [], [], []
    for frame in manifest['frames']:
        fid = frame['frame_id']; geom = f'geometry/frames/{fid}'
        verified(frame['input'])
        if hashes[frame['input']] != frame['sha256']:
            raise ValueError('Source floor photo SHA disagrees with its frame')
        inputs = {key:f'{geom}/{name}' for key,name in {
            'canonical_rgb_path':'canonical.png','pointmap_path':'pts3d.npy','valid_path':'valid_mask.npy',
            'conf_path':'conf.npy','K_path':'intrinsics.npy','c2w_path':'camera_to_world.npy'}.items()}
        for path in inputs.values():verified(path)
        points = np.load(root/inputs['pointmap_path'], allow_pickle=False)
        valid = np.load(root/inputs['valid_path'], allow_pickle=False).astype(bool)
        confidence = np.load(root/inputs['conf_path'], allow_pickle=False)
        C = np.load(root/inputs['c2w_path'], allow_pickle=False)
        if points.shape != (*valid.shape,3) or confidence.shape != valid.shape or C.shape != (4,4):
            raise ValueError('Floor source pointmap/camera grid mismatch: '+fid)
        _, keep = camera_depth(points,C,valid)
        keep &= np.isfinite(confidence) & (confidence >= .1)
        if frame.get('content_rect_xyxy'):
            x0,y0,x1,y1 = frame['content_rect_xyxy']
            content = np.zeros(valid.shape,bool);content[y0:y1,x0:x1] = True;keep &= content
        mask = keep & (np.abs(points@plane[:3]+plane[3]) < threshold)
        cameras.append(C)
        if not mask.any():continue
        directory = root/'evidence/observed-floor'/fid;directory.mkdir(parents=True,exist_ok=True)
        np.save(directory/'canonical_mask.npy',mask)
        cloud = points[mask];clouds.append(cloud)
        np.save(directory/'points.npy',cloud)
        views.append({**inputs,'frame_id':fid,'rgb_path':frame['input'],
            'canonical_mask_path':(directory/'canonical_mask.npy').relative_to(root).as_posix(),
            'points_path':(directory/'points.npy').relative_to(root).as_posix(),
            'mask_pixels':int(mask.sum()),'partial_point_count':len(cloud),'observed_only':True,
            'sha256':{p.name:digest(p) for p in directory.iterdir()},
            'provenance':{'source_run_id':manifest['source_run_id'],'source_image_sha256':frame['sha256'],
                'source_scene_sha256':source['sha256'],'source_geometry_sha256':{p:hashes[p] for p in inputs.values()},
                'mask_resolution':'canonical geometry grid; not a semantic mask or native-resolution segmentation',
                'method':'valid positive-depth pixels within the saved native floor residual gate'}})
    if len(views) < min(2,len(manifest['frames'])) or sum(map(len,clouds)) < 200:
        raise ValueError('Saved floor requires 200 actual support pixels across the available source views (up to two)')
    points = np.concatenate(clouds)
    centers = np.array([c[:3,3] for c in cameras])
    up = np.mean([-c[:3,1] for c in cameras],axis=0)
    if np.linalg.norm(up)<1e-9 or plane[:3]@(up/np.linalg.norm(up)) < _FLOOR_NORMAL_MAX_ANGLE_COS:
        raise ValueError('Saved floor normal is inconsistent with the source camera-up prior')
    heights = centers@plane[:3]+plane[3]
    if not np.isfinite(heights).all() or np.any(heights<=0):
        raise ValueError('Saved floor is not below every source camera')
    _,singular,axes = np.linalg.svd(points-points.mean(0),full_matrices=False)
    if singular[1] <= 1e-12 or abs(axes[-1]@plane[:3]) < _FLOOR_NORMAL_MAX_ANGLE_COS:
        raise ValueError('Saved floor support is degenerate or inconsistent with its plane')
    residual = np.abs(points@plane[:3]+plane[3])
    # The assembler displays one observed frame; choose greatest real support.
    views.sort(key=lambda view:(view['mask_pixels'],view['frame_id']))
    result = {'plane_native':plane.tolist(),'up_native':plane[:3].tolist(),'views':views,
        'metric_scale_known':False,'fit':'saved product native floor; support revalidated against exact source depth',
        'source_scene_sha256':source['sha256'],'distance_threshold_native':threshold,
        'threshold_source':'geometry._fit_floor final 0.03 / saved scale_factor; no new calibration',
        'total_observed_points':len(points),'inlier_points':len(points),
        'all_point_residual_p95_native':float(np.quantile(residual,.95)),
        'camera_signed_heights_native':heights.tolist(),
        'evidence_type':'estimated observed depth support; floor identity and physical units are not independently verified'}
    path = root/'evidence/floor.json'
    if path.exists():raise FileExistsError('Preserve an existing floor fit')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='views'}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ehs-repo', type=Path, required=True)
    args=parser.parse_args()
    try:
        prepare(args.root.resolve(),args.ehs_repo.resolve())
    except (ValueError, OSError, KeyError) as error:
        path=args.root/'evidence/floor-validation.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'state':'blocked','reason':str(error)},indent=2)+'\n')
        raise
