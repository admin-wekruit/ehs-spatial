"""Fill only explicitly missing RGB body predictions; reuse paid cache unchanged."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import urllib.request
import cv2

from build_video_body_models import save_inputs
from reconstruct_room_rgb import digest
from modal_apps.sam3_video_fal import execute


def run(args):
    scene=json.loads(args.scene.read_text());evidence=json.loads(args.evidence.read_text());analysis=json.loads(args.analysis.read_text())
    if scene['source_video_sha256']!=digest(args.video) or scene['provenance']['analysis_sha256']!=digest(args.analysis) or scene['provenance']['evidence_sha256']!=digest(args.evidence):
        raise ValueError('Gap evidence, video or masks differ')
    gaps={(x['sourceFrame'],x['entityId']) for x in evidence['bodies'] if x['status']=='no_supported_cached_body'}
    if len(gaps)>args.max_calls or len(gaps)*.02>args.max_usd:raise ValueError('Missing predictions exceed bounded budget')
    args.output.mkdir(parents=True,exist_ok=False)
    reused=0
    for source in sorted(args.reuse_run.glob('frame-*')):
        if not (source/'provider-mesh.ply').exists():continue
        folder=args.output/source.name;folder.mkdir()
        for name in ['input-manifest.json','image.png','mask.png','provider-output.json','provider-mesh.ply']:
            os.link(source/name,folder/name)
        (folder/'reused-from.json').write_text(json.dumps({'source':str(source.resolve()),'new_provider_submission':False}))
        reused+=1
    frames={f['sourceFrame']:f for f in analysis['frames']};jobs=[]
    cap=cv2.VideoCapture(str(args.video))
    try:
        for index,entity in sorted(gaps):
            folder=args.output/f'frame-{index:05d}-track-{entity.rsplit("-",1)[-1]}'
            if folder.exists():raise ValueError('Gap report names an existing body prediction')
            folder.mkdir();cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
            if not ok:raise ValueError('Source RGB could not be decoded')
            obj=next(o for o in frames[index]['objects'] if o['entityId']==entity)
            mask=cv2.imread(str(args.analysis.parent/obj['maskUrl']),-1)[:,:,3]>0
            save_inputs(folder,image,mask,{'sourceFrame':index,'entityId':entity,'source_video_sha256':scene['source_video_sha256']})
            jobs.append(folder)
    finally:cap.release()
    plan={'source_scene_sha256':digest(args.scene),'gap_evidence_sha256':digest(args.evidence),'reused':reused,
        'calls':len(jobs),'maximum_usd':args.max_usd,'published_estimate_usd':len(jobs)*.02,'new_training':False}
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan),flush=True)
    if args.prepare_only:return
    quote=None
    def request(folder):
        payload={'mode':'submit','endpoint':'fal-ai/sam-3/3d-body','billing_units':1,'max_fal_usd':.02,
            'input':{**{name+'_url':'data:image/png;base64,'+base64.b64encode((folder/(name+'.png')).read_bytes()).decode() for name in ['image','mask']},
                'export_meshes':True,'include_3d_keypoints':True,'include_mhr_params':True}}
        if quote:payload['batch_pricing_quote']=quote
        execute(payload,folder,'provider-events.jsonl')
        output=json.loads((folder/'provider-output.json').read_text())
        if len(output['meshes'])!=1:raise ValueError('Expected one mask-conditioned body')
        item=output['meshes'][0];urllib.request.urlretrieve(item['url'] if isinstance(item,dict) else item,folder/'provider-mesh.ply')
        return folder.name
    if jobs:
        request(jobs[0])
        for event in map(json.loads,(jobs[0]/'provider-events.jsonl').read_text().splitlines()):
            if event['phase']=='pricing':quote={'pricing':event['data'],'fetched_at':time.time()}
        with ThreadPoolExecutor(max_workers=3) as pool:completed=list(pool.map(request,jobs[1:]))
    (args.output/'complete.json').write_text(json.dumps({**plan,'completed':len(jobs),'actual_billing_reconciled':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','evidence','analysis','video','reuse-run','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-calls',type=int,required=True);p.add_argument('--max-usd',type=float,required=True)
    p.add_argument('--prepare-only',action='store_true');run(p.parse_args())
