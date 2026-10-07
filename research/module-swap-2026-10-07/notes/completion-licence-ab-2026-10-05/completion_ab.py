"""Licence-clean per-object completion A/B on cell 030 (experiment only).

TRELLIS-image-large mesh-only (MIT) and self-hosted SAM 3D Objects (SAM License) against RecGen (TRI non-commercial). Every
mesh is placed by the unchanged assembly (panoptes-serving scripts/research/assemble_lucida_scene.py: assemble() ->
refine(), bounded 9-DOF silhouette + depth render-and-compare on every photo of the object), from a generic initial pose:
  RecGen   its own learned pose (generation/<id>/output.json of run bor1-030-01, as published)
  SAM 3D   its own pose from our pointmap (modal_apps/sam3d_research.py SAM3DObjects, unchanged)
  TRELLIS  no pose: fast_report.r5_bench.align_free (z/y up x 24 yaws, silhouette scale, translation ICP), unchanged
Generation input for TRELLIS and SAM 3D: the photo with the object's largest mask (here photo 1 for all six objects).

  M=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal
  $M run completion_ab.py --stage prepare    # TRELLIS @ pinned HF revision + DINOv2 -> volume panoptes-completion-ab (CPU)
  $M run completion_ab.py --stage trellis    # one A100, block_network: six meshes
  $M run completion_ab.py --stage trellis --variants trellis2v   # the same from every photo's crop (run_multi_image, stochastic)
  $M run completion_ab.py --stage sam3d      # SAM3DObjects (A100), six posed meshes
  AB_RUN=RUN AB_OUT=OUT AB_OBJECTS=a,b AB_FRAME=all $M run completion_ab.py --stage sam3d   # any run; every photo of each object
  SAM3D_BACKEND=http|local|modal ... --stage sam3d   # the candidates through ehs_spatial.providers.sam3d (docs/BACKENDS-v1.md) instead of the class here
  AB_RUN=RUN AB_OUT=OUT AB_OBJECTS=a,b $M run completion_ab.py --stage assemble --variants sam3d,sam3d-frame_0001,...
  $M run completion_ab.py --stage assemble   # CPU: initial poses + assemble() for recgen / trellis / sam3d
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import modal

HERE = Path(__file__).resolve().parent
WT = Path(os.environ.get('PANOPTES_WORKCELL', '/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed'))
SERVING = Path(os.environ.get('PANOPTES_SERVING', '/Users/adam/Desktop/panoptes-public/panoptes-serving'))
CELL = os.environ.get('AB_CELL', '030')  # AB_CELL=090: the September cell (run lucida-replica-01, all nine RecGen objects)
# AB_RUN=DIR AB_OUT=DIR [AB_OBJECTS=a,b,...]: any run in the lucida-replica-01 layout (e.g. a rebuilt geometry), its own output
# directory, and the objects to complete (default: every object of its evidence/objects.json); the 030 / 090 defaults are unchanged
RUN = Path(os.environ['AB_RUN']) if os.environ.get('AB_RUN') else SERVING / 'outputs/candidate-evaluation' / {'030': 'bor1-030-01', '090': 'lucida-replica-01'}[CELL]
OUT = Path(os.environ['AB_OUT']) if os.environ.get('AB_OUT') else Path(
    os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad') + '/checks/'
    + {'030': 'completionAB-out', '090': 'completionAB-090'}[CELL])
OBJECTS = (os.environ['AB_OBJECTS'].split(',') if os.environ.get('AB_OBJECTS') else
           [o['object_id'] for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects']] if os.environ.get('AB_RUN') else
           {'030': ['cart', 'guard', 'left_light_curtain', 'left_post', 'right_light_curtain', 'robot'],
            '090': ['left_light_curtain', 'right_light_curtain', 'left_fence', 'right_fence', 'left_post', 'right_post', 'robot',
                    'cart', 'guard']}[CELL])  # field-checked objects first

TRELLIS_ID, TRELLIS_REV = 'microsoft/TRELLIS-image-large', '25e0d31ffbebe4b5a97464dd851910efc3002d96'
TRELLIS_CODE = '442aa1e1afb9014e80681d3bf604e8d728a86ee7'  # github microsoft/TRELLIS (fast_report.gen3d's pin)
UTILS3D = '9a4eb15e4021b67b12c460c7057d642626897ec8'
DINO_REV, DINO_URL = '7764ea0f912e53c92e82eb78a2a1631e92725fc8', 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'
DINO_SHA = '36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51'  # as scripts/onprem/fetch_weights.py
# the mesh path only (pipeline.json models minus the Gaussian and radiance-field decoders); HF LFS sha256 at TRELLIS_REV
MESH_MODELS = {'sparse_structure_decoder': ('ss_dec_conv3d_16l8_fp16', '1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a'),
               'sparse_structure_flow_model': ('ss_flow_img_dit_L_16l8_fp16', '96dc6bfd4136fd950af564dd16b4ae533c9ba6af8f26c670646b2a9f2789b1db'),
               'slat_decoder_mesh': ('slat_dec_mesh_swin8_B_64l8m256c_fp16', '3e87aba94b5786407eb06d0502c1ed0885a0027a3f2b8537bfe15b0a92c01859'),
               'slat_flow_model': ('slat_flow_img_dit_L_64l8p2_fp16', '693fb2a58ad497bd222007301eeec49d14d60f8c12d2f2f00c221fa747b4c66c')}
NOT_LICENCE_CLEAN = ['nvdiffrast', 'diff_gaussian_rasterization', 'diffoctreerast', 'kaolin', 'rembg', 'pymeshfix', 'xatlas',
                     'pyvista', 'igraph', 'bpy', 'smplx', 'recgen_inference']
A100 = .000694 + 4 * .0000131 + 32 * .00000222   # USD/s list price, A100-80GB + 4 CPU + 32 GiB (SAM 3D's container)
CPU8 = 8 * .0000131 + 16 * .00000222

app = modal.App('panoptes-completion-ab')
vol = modal.Volume.from_name('panoptes-completion-ab', create_if_missing=True)
dl_image = modal.Image.debian_slim(python_version='3.11').pip_install('huggingface_hub==0.36.0')
# RecGen's image prefix (modal_apps/lucida_assets.py, same pins as fast_report.gen3d's TRELLIS venv), then TRELLIS instead of RecGen
trellis_image = (modal.Image.from_registry('nvidia/cuda:12.1.1-devel-ubuntu22.04', add_python='3.10')
    .apt_install('git', 'build-essential', 'ninja-build', 'libgl1', 'libglib2.0-0')
    .env({'TORCH_CUDA_ARCH_LIST': '8.0', 'MAX_JOBS': '4', 'ATTN_BACKEND': 'xformers', 'PYTHONPATH': '/opt/recgen',
          'SPCONV_ALGO': 'native', 'HF_HUB_DISABLE_TELEMETRY': '1'})
    .pip_install('torch==2.4.0', 'torchvision==0.19.0', index_url='https://download.pytorch.org/whl/cu121')
    .pip_install('numpy==1.26.4', 'pillow==11.3.0', 'scipy==1.15.3', 'easydict==1.13', 'tqdm==4.67.1', 'safetensors==0.6.2',
                 'huggingface_hub==0.36.0', 'spconv-cu120==2.3.6', 'trimesh==4.7.4', 'plyfile==1.1.2', 'einops==0.8.1',
                 'opencv-python-headless==4.11.0.86')
    .pip_install('xformers==0.0.27.post2', index_url='https://download.pytorch.org/whl/cu121')
    .run_commands(f'pip install git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D}',
                  f'git clone https://github.com/microsoft/TRELLIS.git /opt/trellis && git -C /opt/trellis checkout --detach {TRELLIS_CODE} '
                  '&& git -C /opt/trellis submodule update --init trellis/representations/mesh/flexicubes',
                  f'git clone https://github.com/facebookresearch/dinov2.git /opt/dinov2 && git -C /opt/dinov2 checkout --detach {DINO_REV}')
    .env({'PYTHONPATH': '/opt/fr', 'HF_HUB_OFFLINE': '1'})
    .add_local_file(WT / 'fast_report/__init__.py', '/opt/fr/fast_report/__init__.py')
    .add_local_file(WT / 'fast_report/gen3d.py', '/opt/fr/fast_report/gen3d.py'))
# modal_apps/assemble_scene.py's image, plus fast_report's align_free
cpu_image = (modal.Image.debian_slim(python_version='3.12')
    .apt_install('libgl1', 'libgomp1', 'libglib2.0-0', 'libx11-6')
    .pip_install('numpy==2.1.3', 'opencv-python-headless==4.10.0.84', 'open3d==0.19.0', 'pillow==11.0.0', 'scipy==1.14.1', 'trimesh==4.4.9')
    .add_local_dir(SERVING / 'scripts/research', '/serving/scripts/research', ignore=['__pycache__'])
    .add_local_file(WT / 'fast_report/__init__.py', '/opt/fr/fast_report/__init__.py')
    .add_local_file(WT / 'fast_report/r5_bench.py', '/opt/fr/fast_report/r5_bench.py')
    .add_local_file(WT / 'fast_report/x7.py', '/opt/fr/fast_report/x7.py'))

if modal.is_local():
    sys.path.insert(0, str(WT / 'modal_apps'))
    import sam3d_research  # noqa: E402  its SAM3DObjects class, unchanged
    app.include(sam3d_research.app)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            h.update(block)
    return h.hexdigest()


@app.function(image=dl_image, volumes={'/w': vol}, cpu=2, memory=4096, timeout=1800, retries=0, min_containers=0)
def prepare_weights():
    import urllib.request
    from huggingface_hub import hf_hub_download
    root = Path('/w/trellis-image-large')
    names = ['pipeline.json', 'README.md'] + [f'ckpts/{n}.{e}' for n, _ in MESH_MODELS.values() for e in ('json', 'safetensors')]
    for name in names:
        hf_hub_download(TRELLIS_ID, name, revision=TRELLIS_REV, local_dir=str(root))
    for name, sha in MESH_MODELS.values():
        assert digest(root / f'ckpts/{name}.safetensors') == sha, name
    config = json.loads((root / 'pipeline.json').read_text())
    config['args']['models'] = {k: f'../trellis-image-large/ckpts/{n}' for k, (n, _) in MESH_MODELS.items()}
    mesh = Path('/w/trellis-mesh')
    mesh.mkdir(exist_ok=True)
    (mesh / 'pipeline.json').write_text(json.dumps(config, indent=2))
    dino = Path('/w/dinov2_vitl14_reg4_pretrain.pth')
    if not dino.exists():
        urllib.request.urlretrieve(DINO_URL, dino.with_suffix('.part'))
        dino.with_suffix('.part').rename(dino)
    assert digest(dino) == DINO_SHA
    files = [f'trellis-image-large/{n}' for n in names] + ['trellis-mesh/pipeline.json', dino.name]
    manifest = {'trellis': {'model': TRELLIS_ID, 'revision': TRELLIS_REV, 'code': TRELLIS_CODE, 'licence': 'MIT'},
                'dinov2': {'url': DINO_URL, 'code': DINO_REV, 'licence': 'Apache-2.0'},
                'mesh_pipeline_models': config['args']['models'],
                'sha256': {f: digest(Path('/w') / f) for f in files}}
    Path('/w/manifest.json').write_text(json.dumps(manifest, indent=2))
    vol.commit()
    return manifest


@app.function(image=trellis_image, gpu='A100-80GB', cpu=4, memory=32768, volumes={'/w': vol}, timeout=1500, retries=0,
              min_containers=0, max_containers=1, block_network=True, single_use_containers=True)
def trellis_meshes(job: str, objects: dict, seed: int = 42, cells: int = 256):
    """Official pipeline defaults (25 + 25 steps, cfg 5); official preprocess_image on our RGBA crop (its alpha = our mask,
    so rembg is never called). Mesh decoder only; vertex clustering at cells^3 as fast_report.gen3d."""
    import importlib.metadata
    import numpy as np
    import torch
    from PIL import Image
    from fast_report import gen3d
    started = time.monotonic()
    vol.reload()
    manifest = json.loads(Path('/w/manifest.json').read_text())
    for rel, sha in manifest['sha256'].items():
        assert digest(Path('/w') / rel) == sha, rel
    gen3d.TRELLIS_PIPE, gen3d.DINO_WEIGHTS = '/w/trellis-mesh', '/w/dinov2_vitl14_reg4_pretrain.pth'
    pipe = gen3d.load_trellis()
    loaded = time.monotonic() - started
    out = Path(f'/w/jobs/{job}/out')
    out.mkdir(parents=True)
    record = {'manifest': manifest, 'seed': seed, 'cluster_cells': cells, 'load_seconds': loaded, 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'sampler_params': {'sparse_structure': pipe.sparse_structure_sampler_params,
                                                             'slat': pipe.slat_sampler_params}, 'objects': {}}
    for oid, names in objects.items():  # one crop: run(); several (largest mask first): run_multi_image, stochastic
        images = [Image.open(f'/w/jobs/{job}/{name}') for name in names]
        for image in images:
            image.load()
            assert image.mode == 'RGBA'
        torch.cuda.reset_peak_memory_stats()
        t = time.monotonic()
        result = (pipe.run(images[0], seed=seed, formats=['mesh']) if len(images) == 1 else
                  pipe.run_multi_image(images, seed=seed, formats=['mesh'], mode='stochastic'))
        torch.cuda.synchronize()
        seconds = time.monotonic() - t
        m = result['mesh'][0]
        V, F = m.vertices.float().cpu().numpy(), m.faces.cpu().numpy()
        Vc, Fc = gen3d.cluster(V, F, cells)
        np.savez_compressed(out / f'{oid}.npz', vertices=Vc.astype(np.float32), faces=Fc.astype(np.int32))
        record['objects'][oid] = {'seconds': seconds, 'faces_raw': int(len(F)), 'vertices': int(len(Vc)), 'faces': int(len(Fc)),
                                  'bounds': [Vc.min(0).tolist(), Vc.max(0).tolist()], 'input_sizes': [list(i.size) for i in images], 'inputs': names,
                                  'peak_cuda_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
    dists = {d.metadata['Name'].lower().replace('-', '_'): d.version for d in importlib.metadata.distributions()}
    record['licence_audit'] = {
        'imported_real': sorted(n for n in NOT_LICENCE_CLEAN if getattr(sys.modules.get(n), '__file__', None)),
        'stubbed_never_called': sorted(n for n in NOT_LICENCE_CLEAN if n in sys.modules and not getattr(sys.modules[n], '__file__', None)),
        'installed': sorted(n for n in NOT_LICENCE_CLEAN if n.replace('-', '_') in dists),
        'distributions': dict(sorted(dists.items()))}
    record['function_seconds'] = time.monotonic() - started
    (out / 'record.json').write_text(json.dumps(record, indent=1))
    vol.commit()
    return {'dir': f'jobs/{job}/out', 'files': {p.name: digest(p) for p in out.iterdir()}}


def best_view(obj, frame_id=None):
    """The generation photo: the largest mask, or the given frame (variant 'sam3d-<frame_id>': SAM 3D from another photo)."""
    if frame_id:
        return next(v for v in obj['views'] if v['frame_id'] == frame_id)
    return max(obj['views'], key=lambda v: v['mask_pixels'])


def variant_frame(variant):
    return variant.split('-', 1)[1] if variant.startswith('sam3d-') else None


def sam3d_inputs(obj, manifest, factor=2, frame_id=None):
    """The whole photo (SAM 3D crops around the mask itself and infers a centred-principal-point K from the pointmap) at
    1/factor, the accepted original-resolution mask, and our Pi3X depth (nearest canonical pixel) as its pointmap:
    PyTorch3D camera (OpenCV x, y negated), NaN = no depth (as complete_video_objects.sam3d_pointmap)."""
    import cv2
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(SERVING / 'scripts/research'))
    from generate_lucida_assets import camera_depth
    view = best_view(obj, frame_id)
    frame = next(f for f in manifest['frames'] if f['frame_id'] == view['frame_id'])
    assert digest(RUN / frame['input']) == frame['sha256']
    photo = Image.open(RUN / frame['input']).convert('RGB')
    W, H = photo.size
    w, h = W // factor, H // factor
    rgb = np.asarray(photo.resize((w, h), Image.Resampling.BOX))
    mask = cv2.resize((np.asarray(Image.open(RUN / view['mask_path'])) > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= .5
    points = np.load(RUN / view['pointmap_path'])
    c2w = np.load(RUN / view['c2w_path'])
    K = np.load(RUN / view['K_path']).astype(np.float64)
    conf = np.load(RUN / view['conf_path'])
    valid = np.load(RUN / view['content_valid_path']).astype(bool) & np.isfinite(conf) & (conf >= .1)
    depth, _ = camera_depth(points, c2w, valid)
    A = np.asarray(frame['input_to_canonical_pixel_centres'], float)
    v, u = np.indices((h, w), dtype=np.float64)
    uf, vf = factor * u + (factor - 1) / 2, factor * v + (factor - 1) / 2  # full-resolution pixel centres
    cx = np.floor(A[0, 0] * uf + A[0, 2] + .5).astype(int)
    cy = np.floor(A[1, 1] * vf + A[1, 2] + .5).astype(int)
    inside = (cx >= 0) & (cy >= 0) & (cx < depth.shape[1]) & (cy < depth.shape[0])
    z = np.full((h, w), np.nan, np.float32)
    z[inside] = depth[cy[inside], cx[inside]]
    z[z <= 0] = np.nan
    Kf = np.linalg.inv(A) @ K  # full-resolution K
    Ks = np.diag([1 / factor, 1 / factor, 1.]) @ Kf
    Ks[:2, 2] = (Kf[:2, 2] - (factor - 1) / 2) / factor
    pointmap = np.stack([-(u - Ks[0, 2]) / Ks[0, 0] * z, -(v - Ks[1, 2]) / Ks[1, 1] * z, z], -1).astype(np.float32)
    meta = {'frame_id': view['frame_id'], 'grid_hw': [h, w], 'factor': factor, 'K': Ks.tolist(), 'mask_pixels': int(mask.sum()),
            'mask_depth_pixels': int((mask & np.isfinite(z)).sum())}
    return rgb, mask, pointmap, meta


@app.function(image=cpu_image, cpu=8, memory=16 * 1024, timeout=3600, retries=0, min_containers=0)
def assemble_variants(archive: bytes, variants: list, iterations: int = 100, eval_size: int = 288):
    import shutil
    import numpy as np
    import trimesh
    sys.path[:0] = ['/serving/scripts/research', '/opt/fr']
    import assemble_lucida_scene as als
    import fast_report.x7 as x7m
    from fast_report.r5_bench import align_free
    started = time.monotonic()
    work = Path('/tmp/w')
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        bundle.extractall(work, filter='data')
    shared = work / 'shared'
    objects = {o['object_id']: o for o in json.loads((shared / 'evidence/objects.json').read_text())['objects']}
    up = np.asarray(json.loads((shared / 'evidence/floor.json').read_text())['up_native'], float)
    results = {}
    for variant in variants:
        root = Path('/tmp/runs') / variant
        (root / 'generation').mkdir(parents=True)
        for name in ['manifest.json', 'evidence', 'geometry']:
            (root / name).symlink_to(shared / name)
        inits = {}
        for gen in sorted((work / 'gen' / variant).iterdir()):
            oid, out = gen.name, root / 'generation' / gen.name
            if variant == 'recgen':
                shutil.copytree(gen, out)
                continue
            obj = objects[oid]
            data = np.load(gen / 'mesh.npz')
            V, F = data['vertices'].astype(np.float64), data['faces'].astype(np.int64)
            anchor = als.native_view(root, best_view(obj, variant_frame(variant)))
            if variant.startswith('trellis'):  # no pose from the model: the generic upright yaw search, then assemble's refine
                views = sorted(obj['views'], key=lambda v: -v['mask_pixels'])
                O = []
                for spec in views:
                    nv = als.native_view(root, spec)
                    O.append(nv['points'][np.load(root / spec['canonical_mask_path']).astype(bool) & nv['valid']])
                O = np.concatenate(O)
                O = O[np.random.default_rng(0).choice(len(O), min(len(O), 8000), replace=False)]
                amask = np.load(root / views[0]['canonical_mask_path']).astype(bool)
                saved = x7m.observed_points, x7m.view_data
                x7m.observed_points = lambda src, key, frames, cap=60000: O
                x7m.view_data = lambda src, key, frame, scale=1.: (None, amask, anchor['K'], anchor['c2w'])
                try:
                    T, info = align_free(None, oid, [views[0]['frame_id']], up, {'vertices': V, 'faces': F})
                finally:
                    x7m.observed_points, x7m.view_data = saved
                object_to_camera = np.linalg.inv(anchor['c2w']) @ T
            else:  # SAM 3D: its pose puts the mesh in the PyTorch3D camera of our pointmap (x, y negated)
                object_to_camera = np.diag([-1., -1, 1, 1]) @ data['object_to_camera_p3d'].astype(np.float64)
                info = {'pose': 'SAM 3D from our pointmap'}
            als.decompose(anchor['c2w'] @ object_to_camera)
            out.mkdir()
            trimesh.Trimesh(V, F, process=False).export(out / 'object.ply')
            trimesh.Trimesh(als.transformed(V, object_to_camera), F, process=False).export(out / 'posed-object.ply')
            model = {'trellis': TRELLIS_ID + '@' + TRELLIS_REV, 'trellis2v': TRELLIS_ID + '@' + TRELLIS_REV + ' (multi-image)', 'sam3d': 'facebook/sam-3d-objects'}[variant.split('-')[0]]
            record = {'status': 'complete', 'object_id': oid, 'model_id': model, 'anchor_frame': anchor['frame_id'],
                      'object_to_camera': object_to_camera.tolist(), 'init': info,
                      'paths': {'mesh': 'object.ply', 'posed_mesh': 'posed-object.ply'},
                      'output_sha256': {n: digest(out / n) for n in ['object.ply', 'posed-object.ply']}}
            (out / 'output.json').write_text(json.dumps(record, indent=1))
            inits[oid] = info
        t = time.monotonic()
        als.assemble(root, iterations, eval_size)
        comparisons = json.loads((root / 'result/comparisons.json').read_text())
        silhouettes = {}
        for c in comparisons['objects']:
            c['refinement'].pop('trajectory', None)
            obj, rec = objects[c['object_id']], json.loads((root / 'generation' / c['object_id'] / 'output.json').read_text())
            mesh = trimesh.load(root / 'generation' / c['object_id'] / rec['paths']['mesh'], force='mesh', process=False)
            for spec in obj['views']:
                view = als.load_view(root, spec, c['evaluation_grid_height'])
                for key in ['initial_object_to_world', 'final_object_to_world']:
                    hit = als.cast_depth(mesh, np.asarray(c[key]), view['rays'])
                    silhouettes[f"{c['object_id']}/{spec['frame_id']}/{key[:-16]}"] = np.packbits(np.isfinite(hit) & (hit > 0)).tobytes()
                silhouettes[f"{c['object_id']}/{spec['frame_id']}/target"] = np.packbits(view['target']).tobytes()
                silhouettes[f"{c['object_id']}/{spec['frame_id']}/shape"] = json.dumps(view['target'].shape).encode()
        results[variant] = {'comparisons': comparisons, 'inits': inits, 'assembly_seconds': time.monotonic() - t,
                            'silhouettes': silhouettes}
    return {'results': results, 'container_seconds': time.monotonic() - started}


def stage_trellis(variant):
    import uuid
    objects = {o['object_id']: o for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects']}
    job = uuid.uuid4().hex
    crops = {}
    with vol.batch_upload() as upload:
        for oid in OBJECTS:
            views = sorted(objects[oid]['views'], key=lambda v: -v['mask_pixels'])[:None if variant == 'trellis2v' else 1]
            for v in views:
                upload.put_file(RUN / v['rgba_path'], f"/jobs/{job}/{oid}-{v['frame_id']}.png")
            crops[oid] = [f"{oid}-{v['frame_id']}.png" for v in views]
    t = time.monotonic()
    manifest = trellis_meshes.remote(job, crops)
    call = time.monotonic() - t
    dest = OUT / variant
    dest.mkdir(parents=True)
    for name, sha in manifest['files'].items():
        data = b''.join(vol.read_file(f"/{manifest['dir']}/{name}"))
        assert hashlib.sha256(data).hexdigest() == sha, name
        (dest / name).write_bytes(data)
    record = json.loads((dest / 'record.json').read_text())
    record['call_seconds'] = call
    record['estimate_usd'] = record['function_seconds'] * A100
    (dest / 'record.json').write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k not in ('manifest',)}, default=str)[:3000])


def stage_sam3d(frame_id=None):
    import numpy as np
    manifest = json.loads((RUN / 'manifest.json').read_text())
    if frame_id == 'all':  # AB_FRAME=all: the largest-mask photo, then every other photo, in one app run (one warm container)
        for f in [None] + [f['frame_id'] for f in manifest['frames']]:
            stage_sam3d(f)
        return
    objects = {o['object_id']: o for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects']}
    dest = OUT / ('sam3d-' + frame_id if frame_id else 'sam3d')
    dest.mkdir(parents=True, exist_ok=True)
    backend = os.environ.get('SAM3D_BACKEND')  # set: ehs_spatial.providers.sam3d (http service | local | modal); unset: the Modal class as before
    if backend:
        sys.path.insert(0, str(SERVING))
        from ehs_spatial.providers import sam3d as sam3d_provider
    model = None if backend else sam3d_research.SAM3DObjects()
    log = json.loads((dest / 'record.json').read_text()) if (dest / 'record.json').exists() else {'objects': {}}
    for oid in OBJECTS:
        if (dest / f'{oid}.npz').exists():
            continue
        if frame_id and (frame_id not in [v['frame_id'] for v in objects[oid]['views']] or best_view(objects[oid])['frame_id'] == frame_id):
            continue  # no such photo, or it is the default generation photo (already in sam3d/)
        rgb, mask, pointmap, meta = sam3d_inputs(objects[oid], manifest, frame_id=frame_id)
        t = time.monotonic()
        out = sam3d_provider.generate(rgb, mask, pointmap, 42) if backend else model.run.remote(rgb, mask, pointmap, 42)
        meta['call_seconds'] = time.monotonic() - t
        if 'error' in out:
            meta.update(error=out['error'][-3000:], seconds=out['seconds'])
            print(oid, 'error', out['error'][-800:])
        else:
            np.savez_compressed(dest / f'{oid}.npz', vertices=out['vertices'], faces=out['faces'].astype(np.int32), colors=out['colors'],
                                object_to_camera_p3d=out['object_to_camera_p3d' if backend else 'objectToCamera'])
            meta.update(seconds=out['seconds'], gpu=out['gpu'], pins=out['pins'], vertices=len(out['vertices']), faces=len(out['faces']))
            if backend:
                meta.update(backend=backend, model_info=out.get('model_info'))
            print(oid, len(out['faces']), 'faces', round(out['seconds'], 1), 's')
        log['objects'][oid] = meta
        (dest / 'record.json').write_text(json.dumps(log, indent=1))


def stage_assemble(variants):
    import numpy as np
    buffer = io.BytesIO()
    objects = json.loads((RUN / 'evidence/objects.json').read_text())['objects']
    # dereference: a run may hold symlinks (e.g. masks linked to the source run); the container's extractall(filter='data')
    # refuses absolute links, so the archive carries the files themselves
    with tarfile.open(fileobj=buffer, mode='w:gz', dereference=True) as bundle:
        for rel in ['manifest.json', 'evidence/objects.json', 'evidence/floor.json', 'geometry/frames']:
            bundle.add(RUN / rel, arcname=f'shared/{rel}')
        for p in sorted((RUN / 'evidence/objects').glob('*/*/canonical_mask.npy')):
            bundle.add(p, arcname=f'shared/{p.relative_to(RUN)}')
        for oid in [o['object_id'] for o in objects] if 'recgen' in variants else []:
            for name in ['object.ply', 'posed-object.ply', 'output.json']:
                bundle.add(RUN / 'generation' / oid / name, arcname=f'gen/recgen/{oid}/{name}')
        for variant in [v for v in variants if v != 'recgen']:
            for oid in OBJECTS:
                src = OUT / variant / f'{oid}.npz'
                if src.exists():
                    if variant.startswith('sam3d'):  # mesh.npz with the keys assemble_variants reads
                        d = np.load(src)
                        one = io.BytesIO()
                        np.savez(one, vertices=d['vertices'], faces=d['faces'], object_to_camera_p3d=d['object_to_camera_p3d'])
                        info = tarfile.TarInfo(f'gen/{variant}/{oid}/mesh.npz')
                        info.size = one.getbuffer().nbytes
                        one.seek(0)
                        bundle.addfile(info, one)
                    else:
                        bundle.add(src, arcname=f'gen/{variant}/{oid}/mesh.npz')
    print('archive MB', round(buffer.getbuffer().nbytes / 1e6, 1))
    t = time.monotonic()
    result = assemble_variants.remote(buffer.getvalue(), variants)
    call = time.monotonic() - t
    for variant, r in result['results'].items():
        dest = OUT / 'assembly' / variant
        dest.mkdir(parents=True, exist_ok=True)
        sil = r.pop('silhouettes')
        np.savez_compressed(dest / 'silhouettes.npz', **{k.replace('/', '__'): np.frombuffer(v, np.uint8) for k, v in sil.items()})
        (dest / 'comparisons.json').write_text(json.dumps(r, indent=1))
    ledger = {'functionSeconds': result['container_seconds'], 'callSeconds': call, 'estimateUsd': result['container_seconds'] * CPU8}
    (OUT / 'assembly' / f"ledger-{'-'.join(variants)}.json").write_text(json.dumps(ledger, indent=1))
    print(json.dumps(ledger))


@app.local_entrypoint()
def main(stage: str, variants: str = 'recgen,trellis,sam3d'):
    OUT.mkdir(parents=True, exist_ok=True)
    if stage == 'prepare':
        t = time.monotonic()
        manifest = prepare_weights.remote()
        (OUT / 'weights-manifest.json').write_text(json.dumps({**manifest, 'call_seconds': time.monotonic() - t}, indent=1))
        print(json.dumps(manifest, indent=1)[:2000])
    elif stage == 'trellis':
        stage_trellis(variants if variants in ('trellis', 'trellis2v') else 'trellis')
    elif stage == 'sam3d':
        stage_sam3d(os.environ.get('AB_FRAME'))  # AB_FRAME=frame_0002: SAM 3D from that photo instead of the largest mask
    elif stage == 'assemble':
        stage_assemble(variants.split(','))
    else:
        raise ValueError(stage)
