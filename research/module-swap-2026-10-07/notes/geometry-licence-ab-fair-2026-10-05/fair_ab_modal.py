"""Fair geometry A/B on ephemeral Modal functions (nothing deployed, retries=0, min_containers=0).

  python fair_ab_modal.py data                      (local, modal venv) copy the small fixed inputs into SCR/checks/da3fair-data
  modal run fair_ab_modal.py --stage infer          one A100: Pi3X, DA3-LARGE-1.1, DA3-BASE x both cells x {padded, unpadded}
                                                    -> SCR/checks/da3fair-geom/CELL-BACKBONE-VARIANT/geometry
  modal run fair_ab_modal.py --stage analyse        8-CPU map, one job per (cell, backbone, variant) + the published Pi3X
                                                    geometry: fair_ab.py -> SCR/checks/da3fair-analyse/CELL-BACKBONE-VARIANT.json
Runners are the pinned ones in panoptes-serving, unchanged (Pi3XRunner, DA3Runner through MapAnythingAdapter).
"""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

import modal

NOTE = Path(__file__).resolve().parent
SERV = Path(os.environ.get('PANOPTES_SERVING', '/Users/adam/Desktop/panoptes-public/panoptes-serving'))
WT = Path(os.environ.get('PANOPTES_WORKCELL', '/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed'))
CELL030 = Path(os.environ.get('SWAP_NOTES', '/Users/adam/Desktop/panoptes-public/research-notes')) / 'cell030-sept-pipeline-2026-10-05'
CLEARB = Path(os.environ.get('SWAP_NOTES', '/Users/adam/Desktop/panoptes-public/research-notes')) / 'workcell-clearance-b-2026-10-05/clearance_b.py'
GEOMAB = Path(os.environ.get('SWAP_NOTES', '/Users/adam/Desktop/panoptes-public/research-notes')) / 'geometry-licence-ab-2026-10-05/geometry_ab.py'
SCR = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
DATA, GEOM, OUT = SCR / 'checks/da3fair-data', SCR / 'checks/da3fair-geom', SCR / 'checks/da3fair-analyse'
RUNS = SERV / 'outputs/candidate-evaluation'
GPU_RATE = .000694 + 8 * .0000131 + 32 * .00000222  # A100-80GB + 8 CPU + 32 GiB list rate (USD/s); not an invoice
CPU_RATE = 8 * .0000131 + 16 * .00000222
MA_CODE, DA3_CODE, PI3_CODE = '3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9', '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4', '9fa3ddb3f8d53041f8b2738df404f62223bbaa7b'
BACKBONES = {'pi3x': ('yyfz233/Pi3X', 'bb1deea4d7423de5b30691739cb451a3f57dc1d5'),
             'da3-large-1.1': ('depth-anything/DA3-LARGE-1.1', '0e109ae307c5982f319a67cf6f9f99ccdc0ec97c'),
             'da3-base': ('depth-anything/DA3-BASE', 'f4a6c9b3c95e41c82048423d3493a81ec3fa810e')}
X0, XW = 63, 392  # content rect of every canonical frame of both runs: [63, 0, 455, 518]
CELLS = {'090': dict(run='lucida-replica-01', pub='4b58dbd2-3846-47f2-af97-57eaa108753c', view=SCR / 'sept/new-view.json', map=SCR / 'checks/map090.txt'),
         '030': dict(run='bor1-030-01', pub='cd84d3fb-7d1f-4736-8ffa-d44855e59fab', view=SCR / 'checks/cd84-view.json', map=SCR / 'checks/map030.txt')}
DATA.mkdir(parents=True, exist_ok=True)

app = modal.App('geometry-licence-ab-fair')
gpu_image = (modal.Image.debian_slim(python_version='3.11')  # = geometry-licence-ab's image (cached layers) + the Pi3 checkout
             .apt_install('git', 'libgl1', 'libglib2.0-0')
             .pip_install('torch==2.5.1', 'torchvision==0.20.1', 'numpy==1.26.4', 'pillow==11.0.0', 'opencv-python-headless==4.10.0.84',
                          'huggingface_hub==0.36.0', 'safetensors==0.4.5', 'einops==0.8.0', 'trimesh==5.1.0', 'plyfile==1.1', 'pydantic==2.9.2',
                          'addict==2.4.0', 'omegaconf==2.3.0', 'hydra-core==1.3.2', 'imageio==2.36.0', 'tqdm==4.67.1', 'scipy==1.14.1',
                          'jaxtyping==0.2.36', 'timm==1.0.11', 'python-box==7.2.0', 'natsort==8.4.0', 'orjson==3.10.11', 'matplotlib==3.9.2',
                          'scikit-learn==1.5.2', 'termcolor==2.5.0')
             .pip_install('uniception==0.1.7', extra_options='--no-deps')
             .run_commands(f'git clone https://github.com/facebookresearch/map-anything.git /vendor/map-anything && git -C /vendor/map-anything checkout {MA_CODE}',
                           f'git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /vendor/depth-anything-3 && git -C /vendor/depth-anything-3 checkout {DA3_CODE}')
             .run_commands(f'git clone https://github.com/yyfz/Pi3.git /vendor/pi3 && git -C /vendor/pi3 checkout {PI3_CODE}')
             .add_local_dir(SERV / 'ehs_spatial', '/serving/ehs_spatial', ignore=['__pycache__'])
             .add_local_file(SERV / 'scripts/candidate_pi3x_backend.py', '/serving/scripts/candidate_pi3x_backend.py')
             .add_local_file(SERV / 'scripts/candidate_geometry_backend.py', '/serving/scripts/candidate_geometry_backend.py'))
cpu_image = (modal.Image.debian_slim(python_version='3.11')
             .apt_install('libgl1', 'libgomp1', 'libx11-6')
             .pip_install('numpy<2.3', 'opencv-python-headless==4.10.0.84', 'open3d==0.19.0', 'trimesh==4.4.9', 'scipy==1.14.1', 'pillow==11.0.0',
                          'shapely==2.0.6', 'pydantic==2.9.2')
             .add_local_dir(SERV / 'ehs_spatial', '/serving/ehs_spatial', ignore=['__pycache__'])
             .add_local_file(WT / 'scripts/workcell_shape_check.py', '/check/shape_core.py')
             .add_local_file(WT / 'scripts/reference_object_scale.py', '/check/reference_object_scale.py')
             .add_local_file(CELL030 / 'estop_cylinder.py', '/check/estop_cylinder.py')
             .add_local_file(CLEARB, '/check/clearance_b.py')
             .add_local_file(GEOMAB, '/check/geometry_ab.py')
             .add_local_file(NOTE / 'fair_ab.py', '/check/fair_ab.py')
             .add_local_dir(DATA, '/data'))


def pack(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        tar.add(root, arcname='.')
    return buf.getvalue()


@app.function(image=gpu_image, gpu='A100-80GB', cpu=8, memory=32 * 1024, timeout=1800, retries=0, min_containers=0)
def infer(inputs: dict) -> dict:
    """inputs = {cell: {variant: {name: png}}}; every backbone sees exactly these bytes."""
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    sys.path[:0] = ['/serving', '/serving/scripts']
    from candidate_geometry_backend import DA3Runner
    from candidate_pi3x_backend import Pi3XRunner
    from ehs_spatial.providers.map_anything import MapAnythingAdapter
    start = time.monotonic(); timing = {}; root = Path('/tmp/out')
    for bb, (repo, rev) in BACKBONES.items():
        t0 = time.monotonic(); wdir = Path('/tmp/w') / bb
        if bb == 'pi3x':
            hf_hub_download(repo, 'model.safetensors', revision=rev, local_dir=wdir)
            runner = Pi3XRunner('/vendor/pi3', wdir, device='cuda')
        else:
            snapshot_download(repo, revision=rev, allow_patterns=['model.safetensors', 'config.json'], local_dir=wdir)
            runner = DA3Runner('/vendor/depth-anything-3', wdir, model_id=repo, device='cuda', process_res=518)
        t1 = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        for cell, variants in inputs.items():
            for variant, frames in variants.items():
                paths = []
                for name, data in sorted(frames.items()):
                    p = Path('/tmp/in') / cell / variant / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data); paths.append(str(p))
                out = root / f'{cell}-{bb}-{variant}' / 'geometry'
                MapAnythingAdapter(runner=runner).run(paths, out)
                meta = json.loads(json.dumps(runner.metadata, default=str)); meta['gpu'] = torch.cuda.get_device_name(0); meta['inputVariant'] = variant
                (out / 'candidate_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
                for junk in ('provider', 'map_anything_response.json', 'point_cloud.glb'):  # raw copies; frames/ keeps every array
                    p = out / junk; shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
        timing[bb] = dict(loadSeconds=t1 - t0, inferSeconds=time.monotonic() - t1, peakGiB=torch.cuda.max_memory_allocated() / 2 ** 30)
        del runner; torch.cuda.empty_cache()
    return dict(archive=pack(root), timing=timing, containerSeconds=time.monotonic() - start)


@app.function(image=gpu_image, cpu=2, memory=4 * 1024, timeout=600, retries=0, min_containers=0)
def freeze() -> dict:
    """pip freeze of the image the DA3 inference ran in (the pins of docker/da3.Dockerfile) + import check of the runner path."""
    import importlib
    import subprocess
    sys.path[:0] = ['/serving', '/serving/scripts', '/vendor/depth-anything-3/src']
    mods = ['candidate_geometry_backend', 'ehs_spatial.providers.map_anything', 'depth_anything_3.cfg', 'depth_anything_3.registry',
            'depth_anything_3.model.da3', 'depth_anything_3.utils.io.input_processor', 'depth_anything_3.utils.io.output_processor']
    loaded = {m: bool(importlib.import_module(m)) for m in mods}
    third = sorted({k.split('.')[0] for k, v in sys.modules.items() if v is not None and getattr(v, '__file__', None) and 'site-packages' in v.__file__})
    return dict(freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True), imports=loaded, sitePackagesImported=third,
                python=sys.version)


@app.function(image=cpu_image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0)
def analyse(job: dict) -> dict:
    sys.path[:0] = ['/check', '/serving']
    import tempfile
    start = time.monotonic()
    import fair_ab as fa
    fa._check()
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(job['geometry']), mode='r:gz') as tar:
            tar.extractall(tmp, filter='data')
        out = analyse_one(job['cell'], job['backbone'], job['variant'], Path(tmp) / 'geometry', Path('/data'))
    out['containerSeconds'] = time.monotonic() - start
    return out


def load_frames(cell: str, geom: Path, data: Path, variant: str):
    import numpy as np
    from PIL import Image
    import fair_ab as fa
    man = json.loads((data / cell / 'manifest.json').read_text()); frames, masks, A = [], [], []
    for f in man['frames']:
        g = geom / 'frames' / f['frame_id']; fid = f['frame_id']
        fr = dict(pts3d=np.load(g / 'pts3d.npy').astype(np.float32), conf=np.load(g / 'conf.npy').astype(np.float32),
                  valid=np.load(g / 'valid_mask.npy').astype(bool), K=np.load(g / 'intrinsics.npy').astype(float),
                  c2w=np.load(g / 'camera_to_world.npy').astype(float))
        canon = np.asarray(Image.open(data / cell / 'canonical' / f'{fid}.png').convert('RGB')); seen = np.asarray(Image.open(g / 'canonical.png').convert('RGB'))
        if variant == 'unpadded':
            assert np.array_equal(seen, canon[:, X0:X0 + XW]), 'not the content crop of the frozen canonical frame'
            fr = fa.embed_unpadded(fr, X0, canon.shape[1])
        else:
            assert np.array_equal(seen, canon), 'not the frozen canonical frame'
        alpha = np.load(data / cell / 'canonical' / f'{fid}_alpha.npy')
        fr['content'] = fa.content_mask(fr['pts3d'], fr['valid'], fr['conf'], alpha)
        m = data / cell / 'floor' / f'{fid}.npy'
        frames.append(fr); masks.append(np.load(m) if m.exists() else None); A.append(np.array(f['input_to_canonical_pixel_centres'], float))
    return frames, masks, A


def analyse_one(cell: str, backbone: str, variant: str, geom: Path, data: Path) -> dict:
    import cv2
    import numpy as np
    import clearance_b as cb
    import fair_ab as fa
    import geometry_ab as ga
    import shape_core as wsc
    from estop_cylinder import fit
    from ehs_spatial.geometry import _ransac_floor_plane
    from reference_object_scale import joint_scale
    frames, masks, A = load_frames(cell, geom, data, variant)
    view = json.loads((data / cell / 'view.json').read_text()); clear = json.loads((data / cell / 'clearb.json').read_text())
    cams_doc = view['cameras']
    cams = [fa.camera(np.linalg.inv(a) @ fr['K'], fr['c2w']) for fr, a in zip(frames, A)]
    sizes = [(c['height'], c['width']) for c in cams_doc]
    out = dict(cell=cell, backbone=backbone, variant=variant, manifest=json.loads((geom / 'candidate_manifest.json').read_text()) if (geom / 'candidate_manifest.json').exists() else None)
    # report-camera check: the Pi3X frames must give the published cameras (same world)
    out['vsReportCameras'] = [dict(KmaxAbs=float(np.abs(np.array(d['K']) - c['K']).max()), c2wMaxAbs=float(np.abs(np.array(d['cameraToWorld']) - c['M']).max())) for d, c in zip(cams_doc, cams)]
    # floor points (frozen masks, content rule) and the three rules
    lowest = ga.floor_fit(frames, masks, _ransac_floor_plane)
    pts = np.concatenate([fr['pts3d'][fr['content'] & m].astype(np.float32) for fr, m in zip(frames, masks) if m is not None])
    centres = np.array([fr['c2w'][:3, 3] for fr in frames]); up = np.mean([-fr['c2w'][:3, 1] for fr in frames], 0); up /= np.linalg.norm(up)
    thr = lowest['thresholdNative']
    rules = fa.floor_rows(pts, centres, up, thr, lowest)
    # masks of the pinned objects (report polygons, original pixels)
    obs = {o['id']: o for o in view['observations']}
    def mask_of(eid, k):
        e = next(x for x in view['entities'] if x['id'].startswith(eid)); m = None
        for oid in e.get('observationRefs') or []:
            o = obs.get(oid)
            if o and o['imageId'] == cams_doc[k]['imageId'] and o.get('originalPixelPolygons'):
                mm = wsc.polygon_mask(o['originalPixelPolygons'], sizes[k]); m = mm if m is None else m | mm
        return m
    pin = fa.PIN[cell]
    post_masks = {(eid, k): mask_of(eid, k - 1) for group in ('housing', 'listed') for eid, ks in pin[group].items() for k in ks}
    # e-stop views (geometry_ab cell_config, unchanged)
    es = json.loads((data / cell / 'estop.json').read_text()); images = {}; views = []
    for v in es['views']:
        K = cams[v['camera']]['K']; K = fa.scaled_K(K, *v['resize']) if v['resize'] else K
        views.append(dict(name=v['name'], K=K, M=cams[v['camera']]['M'], seed=v['seed'], top=v['top'], bot=v['bot']))
        images[v['name']] = cv2.imread(str(data / cell / 'estop' / v['file']), cv2.IMREAD_COLOR)
    P3 = fa.triangulate_axis(views) if es['kind'] == 'triangulate' else fa.pointmap_point(frames[es['frame']], A[es['frame']], es['box'])
    out['floors'] = {}
    for rule, (n, d) in rules.items():
        row = fa.floor_stats(pts, centres, n, d, thr); row['angleToLowestDeg'] = fa.angle_deg(n, rules['lowest'][0])
        e = fa.estop(views, P3, n, fit, joint_scale, images); S = e['nativeToMeters']
        row['estop'] = e; row['gatePassed'] = bool(e['maxDeviation'] < fa.GATE)
        row['cameraHeightsCm'] = [h * S * 100 for h in row['cameraHeightsNative']]
        row['residualP95Cm'] = row['residualP95Native'] * S * 100
        hous = {}
        for group in ('housing', 'listed'):
            for eid, ks in pin[group].items():
                per = {}
                for k in ks:
                    segs = sorted(clear[eid]['photos'][str(k)].get('edgeSegments') or [], key=lambda s: s['medianV'])
                    mask = post_masks[(eid, k)]
                    if not segs or mask is None:
                        per[k] = dict(status='no edge line or mask', heightCm=None); continue
                    per[k] = fa.housing_photo(cams[k - 1], frames[k - 1], A[k - 1], mask, segs[0], (n, d), S, n)  # upper segment = housing edge
                vals = [p['heightCm'] for p in per.values() if p.get('heightCm') is not None]
                vvals = [p['verticalPlaneCm'] for p in per.values() if p.get('verticalPlaneCm') is not None]
                hous[eid] = dict(group=group, photos=per, heightCm=float(np.mean(vals)) if vals else None,
                                 verticalPlaneCm=float(np.mean(vvals)) if vvals else None)
        row['housing'] = hous
        fen = {}
        for eid, ks in pin['fence'].items():
            ph = [clear[eid]['photos'][str(k)] for k in ks]
            fen[eid] = fa.fence_two_view([cams[k - 1] for k in ks], [p['line'] for p in ph], [p['ends'] for p in ph], (n, d), S, cb)
        row['fence'] = fen
        out['floors'][rule] = row
    return out


# ---------------------------------------------------------------- local side
def build_data():
    """Small fixed inputs: run manifests, canonical frames + alphas, frozen floor masks, compact views (cameras, pinned
    entities' polygons), clearance-B edge/fence lines of the pinned objects, e-stop photos and seeds (geometry_ab cell_config)."""
    F = json.loads((SCR / 'estop/estop-features.json').read_text()); N = SERV / 'runs/user-bor1-02/input'
    sept = lambda name, file, cam, resize, key: dict(name=name, file=file, camera=cam, resize=resize, seed=[F[key]['axisX'], (F[key]['headTopRow'] + F[key]['ringBottomRow']) / 2],
                                                     top=F[key]['headTopRow'], bot=F[key]['ringBottomRow'])
    rs = [3024 / 2880, 4032 / 3840]
    estops = {'090': (dict(kind='triangulate', views=[sept('photo 1 (image_03 original)', 'image_03.jpg', 0, rs, 'cam1 image_03'),
                                                       sept('photo 2 (September side)', '2-Photo-2.jpg', 1, None, 'cam2 Sept side'),
                                                       sept('photo 3 (image_04 original)', 'image_04.jpg', 2, rs, 'cam3 image_04')]),
                      {'image_03.jpg': N / 'image_03.jpg', '2-Photo-2.jpg': RUNS / 'lucida-replica-01/input/2-Photo-2.jpg', 'image_04.jpg': N / 'image_04.jpg'}),
              '030': (dict(kind='pointmap', frame=1, box=[[2655, 2756], [2340, 2385]],
                           views=[dict(name='030 photo 2', file='image_02.jpg', camera=1, resize=None, seed=[2705.0, 2350.0], top=2284, bot=2409)]),
                      {'image_02.jpg': RUNS / 'bor1-030-01/input/image_02.jpg'})}
    sys.path.insert(0, str(NOTE)); import fair_ab as fa
    for cell, cf in CELLS.items():
        run = RUNS / cf['run']; d = DATA / cell
        for sub in ('canonical', 'floor', 'estop'):
            (d / sub).mkdir(parents=True, exist_ok=True)
        man = json.loads((run / 'manifest.json').read_text()); shutil.copy(run / 'manifest.json', d / 'manifest.json')
        for f in man['frames']:
            assert f['content_rect_xyxy'] == [X0, 0, X0 + XW, 518], f['content_rect_xyxy']
            shutil.copy(run / f['canonical'], d / 'canonical' / f"{f['frame_id']}.png"); shutil.copy(run / f['alpha'], d / 'canonical' / f"{f['frame_id']}_alpha.npy")
            m = run / 'evidence/objects/observed_floor' / f['frame_id'] / 'canonical_mask.npy'
            if m.exists():
                shutil.copy(m, d / 'floor' / f"{f['frame_id']}.npy")
        doc = json.loads(Path(cf['view']).read_text())['publication']['snapshot']['revision']['document']
        pins = {e for g in ('housing', 'listed', 'fence') for e in fa.PIN[cell][g]}
        ents = [dict(id=e['id'], label=e.get('label'), observationRefs=e.get('observationRefs')) for e in doc['entities'] if e['id'][:8] in pins]
        refs = {r for e in ents for r in e['observationRefs'] or []}
        obs = [dict(id=o['id'], imageId=o['imageId'], originalPixelPolygons=o.get('originalPixelPolygons')) for o in doc['observations'] if o['id'] in refs]
        pairs = [x.split('=', 1) for x in Path(cf['map']).read_text().strip().split(',')]
        assert [c['imageId'] for c in doc['cameras']] == [p[0] for p in pairs], 'camera order = photo map order = frame order'
        (d / 'view.json').write_text(json.dumps(dict(cameras=doc['cameras'], entities=ents, observations=obs)))
        cl = json.loads((SCR / f'checks/clearb-{cell}-b/results.json').read_text())['objects']
        (d / 'clearb.json').write_text(json.dumps({k[:8]: {'photos': {p: {q: v for q, v in ph.items() if q in ('edgeSegments', 'line', 'ends', 'status')}
                                                                    for p, ph in o['photos'].items()}} for k, o in cl.items() if k[:8] in pins}))
        spec, files = estops[cell]; (d / 'estop.json').write_text(json.dumps(spec))
        for name, src in files.items():
            shutil.copy(src, d / 'estop' / name)
    print(json.dumps({c: sorted(str(p.relative_to(DATA)) for p in (DATA / c).rglob('*') if p.is_file()) for c in CELLS}, indent=0)[:3000])


def frames_for(cell):
    from PIL import Image
    run = RUNS / CELLS[cell]['run']; man = json.loads((run / 'manifest.json').read_text()); out = {'padded': {}, 'unpadded': {}}
    for f in man['frames']:
        name = Path(f['canonical']).name; out['padded'][name] = (run / f['canonical']).read_bytes()
        buf = io.BytesIO(); Image.open(run / f['canonical']).convert('RGB').crop((X0, 0, X0 + XW, 518)).save(buf, format='PNG'); out['unpadded'][name] = buf.getvalue()
    return out


def geometry_tar(geom: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for f in sorted((geom / 'frames').iterdir()):
            for name in ('pts3d.npy', 'conf.npy', 'valid_mask.npy', 'intrinsics.npy', 'camera_to_world.npy', 'canonical.png'):
                tar.add(f / name, arcname=f'geometry/frames/{f.name}/{name}')
        if (geom / 'candidate_manifest.json').exists():
            tar.add(geom / 'candidate_manifest.json', arcname='geometry/candidate_manifest.json')
    return buf.getvalue()


@app.local_entrypoint()
def main(stage: str, cells: str = '090,030', only: str = ''):
    if stage == 'freeze':
        r = freeze.remote(); (SCR / 'checks/da3fair-work/gpu-image-freeze.json').write_text(json.dumps(r, indent=1) + '\n')
        print(json.dumps(r['imports']), r['python'].split()[0], len(r['freeze'].splitlines()), 'packages;', 'imported:', ' '.join(r['sitePackagesImported']))
        return
    if stage == 'infer':
        if GEOM.exists():
            raise ValueError(f'{GEOM} exists; choose a fresh one')
        t = time.monotonic(); r = infer.remote({c: frames_for(c) for c in cells.split(',')})
        GEOM.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(r.pop('archive')), mode='r:gz') as tar:
            tar.extractall(GEOM, filter='data')
        ledger = dict(mode='ephemeral modal run', hardware='A100-80GB, 8 CPU, 32 GiB', functionSeconds=r['containerSeconds'], callSeconds=time.monotonic() - t,
                      estimateUsd=GPU_RATE * r['containerSeconds'], timing=r['timing'], rateSource='https://modal.com/pricing')
        (GEOM / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n'); print(json.dumps(ledger))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in cells.split(','):
        jobs.append(dict(cell=c, backbone='pi3x-published', variant='padded', geometry=geometry_tar(RUNS / CELLS[c]['run'] / 'geometry')))
        for bb in BACKBONES:
            for variant in ('padded', 'unpadded'):
                jobs.append(dict(cell=c, backbone=bb, variant=variant, geometry=geometry_tar(GEOM / f'{c}-{bb}-{variant}' / 'geometry')))
    if only:
        jobs = [j for j in jobs if f"{j['cell']}-{j['backbone']}-{j['variant']}" in only.split(',')]
    t = time.monotonic(); total = 0.
    for job, r in zip(jobs, analyse.map(jobs, order_outputs=True, return_exceptions=True)):
        name = f"{job['cell']}-{job['backbone']}-{job['variant']}"
        if isinstance(r, Exception):
            print(name, 'FAILED', repr(r)[:800]); continue
        (OUT / f'{name}.json').write_text(json.dumps(r, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
        total += r['containerSeconds']; print(name, f"{r['containerSeconds']:.0f}s")
    ledger = dict(mode='ephemeral modal run', hardware='8 CPU, 16 GiB per job', jobs=len(jobs), functionSeconds=total, callSeconds=time.monotonic() - t,
                  estimateUsd=CPU_RATE * total, rateSource='https://modal.com/pricing')
    (OUT / f'spend-ledger-{int(time.time())}.json').write_text(json.dumps(ledger, indent=2) + '\n'); print(json.dumps(ledger))


if __name__ == '__main__' and sys.argv[1:] == ['data']:
    build_data()
