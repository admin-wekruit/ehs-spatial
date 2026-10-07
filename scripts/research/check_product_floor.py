"""Zero-model check: floor support is independent of semantic object inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from prepare_product_floor import prepare
from assemble_lucida_scene import load_view, observed_mesh


def check(ehs_repo):
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary).resolve();hashes={};frames=[];expected={}
        def capture(path):
            hashes[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        for number,(height,width) in enumerate([(80,100),(90,120)],1):
            fid=f'frame_{number:04d}';directory=root/'geometry/frames'/fid;directory.mkdir(parents=True)
            yy,xx=np.indices((height,width));K=np.array([[60.,0,width/2],[0,60.,height/4],[0,0,1.]])
            rays=np.stack([(xx-K[0,2])/60,(yy-K[1,2])/60,np.ones_like(xx)],-1)
            floor=yy>height//2
            depth=np.full((height,width),2.);depth[floor]=1.5/rays[floor,1]
            points=rays*depth[...,None]
            confidence=np.ones((height,width));confidence[-12:,-15:]=0
            valid=np.ones((height,width),bool);valid[-8:,:12]=False
            expected[fid]=floor&valid&(confidence>=.1)
            for name,array in {'pts3d':points,'valid_mask':valid,'conf':confidence,
                               'camera_to_world':np.eye(4),'intrinsics':K}.items():
                path=directory/f'{name}.npy';np.save(path,array);capture(path)
            path=directory/'canonical.png';Image.new('RGB',(width,height),(60,100,140)).save(path);capture(path)
            path=root/f'input/image_{number:02d}.png';path.parent.mkdir(exist_ok=True);Image.new('RGB',(width,height),(60,100,140)).save(path);capture(path)
            frames.append({'frame_id':fid,'input':path.relative_to(root).as_posix(),'sha256':hashes[path.relative_to(root).as_posix()],
                           'canonical_shape_hw':[height,width],'content_rect_xyxy':[0,0,width,height]})
        scene=root/'scene.json';scene.write_text(json.dumps({'floor_plane':[0,-1,0,1.5],'scale_factor':1}));capture(scene)
        manifest={'source_run_id':'new-capture','frames':frames,'source_files_sha256':hashes,
                  'source_floor':{'path':'scene.json','sha256':hashes['scene.json']}}
        (root/'manifest.json').write_text(json.dumps(manifest));(root/'evidence').mkdir()
        # No floor candidate, mask or generation is present.
        (root/'evidence/objects.json').write_text(json.dumps({'objects':[{'object_id':'button','label':'button'}]}))
        result=prepare(root,ehs_repo)
        assert len(result['views'])==2 and result['metric_scale_known'] is False
        for view in result['views']:
            assert np.array_equal(np.load(root/view['canonical_mask_path']),expected[view['frame_id']])
            assert len(observed_mesh(load_view(root,view)).faces)>0
        assert result['total_observed_points']==sum(int(m.sum()) for m in expected.values())
        (root/'evidence/floor.json').unlink()
        # A changed input cannot borrow another capture's saved plane.
        original=scene.read_bytes();scene.write_bytes(b'changed')
        try:prepare(root,ehs_repo)
        except ValueError as error:assert 'changed source floor input' in str(error)
        else:raise AssertionError('Changed source accepted')
        scene.write_bytes(original)
        for plane,expected_error in [([0,-1,0,99.],'200 actual support'),([1,0,0,0],'source camera-up')]:
            scene.write_text(json.dumps({'floor_plane':plane,'scale_factor':1}));capture(scene)
            manifest['source_floor']['sha256']=hashes['scene.json'];(root/'manifest.json').write_text(json.dumps(manifest))
            try:prepare(root,ehs_repo)
            except ValueError as error:
                # A wrong vertical plane may be rejected by insufficient support first.
                assert expected_error in str(error) or '200 actual support' in str(error)
            else:raise AssertionError('Unsupported or wrongly oriented floor accepted')
        print('PASS: two different grids, no floor labels, exact supported masks/triangles, changed/missing/wrong plane rejected; zero model calls')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ehs-repo',type=Path,required=True)
    check(parser.parse_args().ehs_repo.resolve())
