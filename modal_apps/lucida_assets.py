"""Bounded private RecGen object inference. No web endpoint or public image host.

Public cache: LUCIDA_PREPARE_ONLY=1 modal run modal_apps/lucida_assets.py --mode prepare
GPU environment check: modal run modal_apps/lucida_assets.py --mode check
Objects: python scripts/research/generate_lucida_assets.py --object-ids left_post
"""
import hashlib
import io
import json
import os
from pathlib import Path
import time
import modal

CODE_REV = 'fe3c9315b439c50ada8b60c12b469d739fd722db'
MODEL_ID = 'TRI-ML/RecGen'
MODEL_REV = 'bc0df7de2e43314830039a35a720731d4c4fac65'
DINO_REV = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINO_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'
MODEL_NAME = 'recgen_base.multiview_stereo'
CACHE = '/cache'
app = modal.App('lucida-private-assets')
volume = modal.Volume.from_name('panoptes-lucida-weights', create_if_missing=True)
cpu_image = modal.Image.debian_slim(python_version='3.10').pip_install('huggingface_hub==0.36.0')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@app.function(image=cpu_image, volumes={CACHE: volume}, cpu=4, memory=8192,
              timeout=1800, max_containers=1, retries=0)
def prepare_weights():
    import urllib.request
    from huggingface_hub import snapshot_download
    destination = Path(CACHE) / 'weights'
    destination.mkdir(exist_ok=True)
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REV,
        cache_dir=str(destination/'hf'), allow_patterns=['pipeline.json', 'ckpts/*', 'README.md',
        'sparse-structure-ft-70k/*', 'slat_denoiser_ema0.9999_step0075000.pt',
        'slat_config.json', 'slat_pose_stats.json']))
    dino = destination / 'dinov2_vitl14_reg4_pretrain.pth'
    if not dino.is_file():
        temporary = dino.with_suffix('.part')
        urllib.request.urlretrieve(DINO_URL, temporary)
        temporary.rename(dino)
    pipeline = json.loads((snapshot/'pipeline.json').read_text())
    assert pipeline['args']['image_cond_model'] == 'dinov2_vitl14_reg'
    files = {str(p.relative_to(snapshot)): {'sha256':digest(p),'bytes':p.stat().st_size}
             for p in sorted(snapshot.rglob('*')) if p.is_file()}
    manifest = {'model_id':MODEL_ID,'model_revision':MODEL_REV,'code_revision':CODE_REV,
        'snapshot':str(snapshot),'files':files,
        'dino':{'path':str(dino),'url':DINO_URL,'sha256':digest(dino),'bytes':dino.stat().st_size,'code_revision':DINO_REV},
        'licenses':{'recgen_code':'Toyota Research Institute Non-Commercial','recgen_weights':'CC-BY-NC-4.0',
                    'trellis_base':'MIT','dinov2':'Apache-2.0','flexicubes':'Apache-2.0'}}
    (Path(CACHE)/'recgen-weights-manifest.json').write_text(json.dumps(manifest,indent=2))
    volume.commit()
    return manifest


if not os.environ.get('LUCIDA_PREPARE_ONLY'):
    gpu_image = (modal.Image.from_registry('nvidia/cuda:12.1.1-devel-ubuntu22.04', add_python='3.10')
        .apt_install('git','build-essential','ninja-build','libgl1','libglib2.0-0')
        .env({'TORCH_CUDA_ARCH_LIST':'8.0','MAX_JOBS':'4','ATTN_BACKEND':'xformers','PYTHONPATH':'/opt/recgen',
              'SPCONV_ALGO':'native','HF_HUB_DISABLE_TELEMETRY':'1'})
        .pip_install('torch==2.4.0','torchvision==0.19.0', index_url='https://download.pytorch.org/whl/cu121')
        .pip_install('numpy==1.26.4','pillow==11.3.0','scipy==1.15.3','easydict==1.13',
                     'tqdm==4.67.1','safetensors==0.6.2','huggingface_hub==0.36.0',
                     'spconv-cu120==2.3.6','trimesh==4.7.4','plyfile==1.1.2','einops==0.8.1',
                     'opencv-python-headless==4.11.0.86')
        .pip_install('xformers==0.0.27.post2', index_url='https://download.pytorch.org/whl/cu121')
        .run_commands('git clone https://github.com/TRI-ML/recgen.git /opt/recgen',
                      f'git -C /opt/recgen checkout --detach {CODE_REV}',
                      'git clone https://github.com/facebookresearch/dinov2.git /opt/dinov2',
                      f'git -C /opt/dinov2 checkout --detach {DINO_REV}',
                      'pip install --no-deps -e /opt/recgen',
                      "python -c 'import xformers.ops as xops; assert xops.fmha.BlockDiagonalMask; from recgen_inference import build_recgen, generate; from recgen_inference.recgen_modules.models.structured_latent_vae import SLatMeshDecoder; print(\"RecGen import ready\")'"))

    @app.function(image=gpu_image, gpu='A100-80GB', cpu=8, memory=65536,
                  volumes={CACHE:volume}, timeout=600, max_containers=1, retries=0,
                  block_network=True, single_use_containers=True)
    def generate_object(job_id: str = '', object_id: str = 'environment-check', seed: int = 42):
        import contextlib
        import traceback
        from unittest.mock import patch
        import numpy as np
        import torch
        import huggingface_hub
        from recgen_inference import build_recgen, generate, generate_multiview

        started = time.monotonic()
        log = io.StringIO()
        record = {'schema_version':1,'input_contract_version':'recgen-input-v2','object_id':object_id,'seed':seed,'status':'failed','code_revision':CODE_REV,
            'model_id':MODEL_ID,'model_revision':MODEL_REV,'model_name':MODEL_NAME,'pose_from_model':True,
            'metric_scale_known':False,'coordinate_space':'native object; posed mesh in anchor OpenCV camera coordinates',
            'texture':'native generated vertex colors; no texture baking or input photo projection',
            'inference_network_blocked':True,'source_payload_sha256':None,
            'loader_adjustments':['Redirect HF file loader to SHA-verified immutable local snapshot',
                'Redirect DINOv2 torch.hub loading to pinned local code and SHA-verified local checkpoint'],
            'inference_settings':{'attention_backend':'official xformers','seed':seed,'posthoc_color':'none',
                                  'mask_erosion_enabled':True,'mask_erosion_params':{'kernel_size':5,'iterations':1},
                                  'quantile_drop_threshold':0.05,'clamp_range':[-2.0,3.0],
                                  'native_crop':{'image_size':518,'aug_size_ratio':1.2,'max_increase_ratio':3.0}}}
        outputs = {}
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                volume.reload()
                if job_id and (len(job_id)!=32 or any(c not in '0123456789abcdef' for c in job_id)):
                    raise ValueError('Invalid private job id')
                payload=(Path(CACHE)/'jobs'/job_id/'input.npz').read_bytes() if job_id else b''
                record['source_payload_sha256']=hashlib.sha256(payload).hexdigest() if payload else None
                mask_erosion_enabled=True
                if payload:
                    arrays=np.load(io.BytesIO(payload),allow_pickle=False)
                    if 'mask_erosion_enabled' in arrays:
                        flag=arrays['mask_erosion_enabled']
                        if flag.shape!=() or flag.dtype!=np.bool_:
                            raise ValueError('mask_erosion_enabled must be a scalar bool')
                        mask_erosion_enabled=bool(flag.item())
                record['inference_settings']['mask_erosion_enabled']=mask_erosion_enabled
                weights=json.loads((Path(CACHE)/'recgen-weights-manifest.json').read_text())
                assert weights['model_revision']==MODEL_REV and weights['code_revision']==CODE_REV
                record['licenses']=weights['licenses']
                assert digest(weights['dino']['path'])==weights['dino']['sha256']
                original_load=torch.hub.load
                used_files=set()
                def pinned_hf(repo_id,filename,*args,**kwargs):
                    if repo_id!=MODEL_ID or args or Path(filename).is_absolute() or '..' in Path(filename).parts:
                        raise ValueError('Unexpected Hugging Face request')
                    path=Path(weights['snapshot'])/filename
                    if filename not in weights['files'] or digest(path)!=weights['files'][filename]['sha256']:
                        raise ValueError('Missing or mismatched immutable weight: '+filename)
                    used_files.add(filename)
                    return str(path)
                def pinned_dino(repo_or_dir,model,*args,**kwargs):
                    if repo_or_dir!='facebookresearch/dinov2' or model!='dinov2_vitl14_reg':
                        raise ValueError('Unexpected torch.hub model')
                    return original_load('/opt/dinov2',model,*args,source='local',
                        weights=weights['dino']['path'],**kwargs)
                with patch('huggingface_hub.hf_hub_download',pinned_hf), patch('torch.hub.load',pinned_dino):
                    pipeline=build_recgen.build(MODEL_NAME)
                torch.cuda.synchronize()
                record['load_seconds']=time.monotonic()-started
                record['hardware']={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda}
                record['weights_manifest_sha256']=digest(Path(CACHE)/'recgen-weights-manifest.json')
                record['used_weight_files']=sorted(used_files)
                if payload:
                    count=int(arrays['view_count'])
                    views=[]
                    for i in range(count):
                        view={key:arrays[f'{i}_{key}'] for key in ['rgb','depth','mask','camera_intrinsics']}
                        if view['rgb'].dtype!=np.uint8 or view['rgb'].shape!=view['depth'].shape+(3,):
                            raise ValueError('RGB/depth grid mismatch')
                        if view['mask'].shape!=view['depth'].shape or not view['mask'].any():
                            raise ValueError('Mask/depth grid mismatch or empty mask')
                        if not np.isfinite(view['depth']).all() or view['depth'].max()>30 or view['depth'].dtype!=np.float32:
                            raise ValueError('Depth must be finite float32 in unscaled native units below upstream auto-mm threshold')
                        if view['camera_intrinsics'].shape!=(3,3) or not np.isfinite(view['camera_intrinsics']).all():
                            raise ValueError('Invalid K')
                        views.append(view)
                    begin=time.monotonic()
                    if count==1:
                        v=views[0]
                        result=generate(pipeline,image=v['rgb'],depth=v['depth'],mask=v['mask'],
                                        intrinsics=v['camera_intrinsics'],seed=seed,mask_erosion_enabled=mask_erosion_enabled)
                    else:
                        result=generate_multiview(pipeline,anchor_view=views[0],second_views=views[1:],seed=seed,
                                                  mask_erosion_enabled=mask_erosion_enabled)
                    torch.cuda.synchronize()
                    record['inference_seconds']=time.monotonic()-begin
                    raw=result.raw_mesh;posed=result.mesh
                    if not len(raw.vertices) or not len(raw.faces) or not np.isfinite(raw.vertices).all():
                        raise ValueError('Native inference did not produce a finite nonempty mesh')
                    object_to_camera=np.linalg.inv(result.cam2ncam.astype(np.float64))@result.pose_matrix
                    predicted=raw.vertices@object_to_camera[:3,:3].T+object_to_camera[:3,3]
                    residual=float(np.max(np.abs(predicted-posed.vertices)))
                    if residual>1e-5:raise ValueError('Native pose composition failed: '+str(residual))
                    for name,mesh in [('object',raw),('posed-object',posed)]:
                        outputs[name+'.ply']=mesh.export(file_type='ply')
                        outputs[name+'.glb']=mesh.export(file_type='glb')
                    record.update(vertices=len(raw.vertices),faces=len(raw.faces),bounds=raw.bounds.tolist(),
                        watertight=bool(raw.is_watertight),body_count=int(raw.body_count),
                        pose_matrix=result.pose_matrix.tolist(),cam2ncam=result.cam2ncam.tolist(),
                        object_to_camera=object_to_camera.tolist(),pose_vertex_max_abs_residual=residual,
                        depth_unit='native supplied pointmap estimated scale, not physically calibrated; supplied float32 unchanged',
                        view_count=count,pose_representation=result.pose_representation,
                        output_sha256={name:hashlib.sha256(data).hexdigest() for name,data in outputs.items()},
                        paths={'mesh':'object.ply','glb':'object.glb','posed_mesh':'posed-object.ply','posed_glb':'posed-object.glb'})
                record['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
                record['status']='complete'
            except Exception:
                record['error']=traceback.format_exc()
                print(record['error'])
        record['gpu_function_seconds']=time.monotonic()-started
        outputs['runtime.log']=log.getvalue().encode()
        outputs['output.json']=(json.dumps(record,indent=2)+'\n').encode()
        import uuid
        remote=Path(CACHE)/'jobs'/(job_id or uuid.uuid4().hex)/'result'
        remote.mkdir(parents=True,exist_ok=False)
        for name,data in outputs.items():(remote/name).write_bytes(data)
        volume.commit()
        return {'volume_path':str(remote.relative_to(CACHE)),
                'files':{name:hashlib.sha256(data).hexdigest() for name,data in outputs.items()}}


def read_outputs(manifest):
    result={}
    for name,sha in manifest['files'].items():
        if Path(name).name!=name:raise ValueError('Invalid returned filename')
        data=b''.join(volume.read_file('/'+manifest['volume_path']+'/'+name))
        if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Private artifact transfer SHA mismatch')
        result[name]=data
    return result


@app.local_entrypoint()
def main(mode: str = 'check', output_dir: str = 'outputs/candidate-evaluation/lucida-replica-01/generation/environment-recgen'):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if mode=='prepare':
        manifest=prepare_weights.remote()
        (output/'weights-manifest.json').write_text(json.dumps(manifest,indent=2))
        print('Public weights ready:',MODEL_ID,MODEL_REV)
    elif mode=='check':
        result=read_outputs(generate_object.remote())
        for name,data in result.items():(output/name).write_bytes(data)
        print((output/'output.json').read_text())
    else:
        raise ValueError('mode must be prepare or check')
