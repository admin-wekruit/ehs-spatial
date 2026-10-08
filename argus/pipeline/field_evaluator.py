"""Pinned DA3-BASE inference and unchanged fair geometry measurement evaluator."""
from argus import ROOT
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time

import modal


def env(key):  # env.template section 6, set by env.sh; a Modal container has none of them and uses nothing read through here
    return os.environ[key] if modal.is_local() else os.environ.get(key, '')


NOTE = Path(__file__).resolve().parent
SERV = ROOT
WT = ROOT
SCR = Path(env('PANOPTES_DATA_ROOT'))
DATA, GEOM, OUT = SCR / 'checks/da3fair-data', SCR / 'checks/da3fair-geom', SCR / 'checks/da3fair-analyse'
RUNS = Path(os.environ.get('PANOPTES_RUNS', SCR / 'runs'))
GPU_RATE = .000694 + 8 * .0000131 + 32 * .00000222  # A100-80GB + 8 CPU + 32 GiB list rate (USD/s); not an invoice
CPU_RATE = 8 * .0000131 + 16 * .00000222
DA3_CODE = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
BACKBONES = {'da3-base': ('depth-anything/DA3-BASE', 'f4a6c9b3c95e41c82048423d3493a81ec3fa810e')}
X0, XW = 63, 392  # content rect of every canonical frame of both runs: [63, 0, 455, 518]
CELLS = {p.stem: json.loads(p.read_text()) for p in (ROOT / 'argus/pipeline/cells').glob('*.json')}

app = modal.App('geometry-licence-ab-fair')
gpu_image = (modal.Image.debian_slim(python_version='3.11')
             .apt_install('git', 'libgl1', 'libglib2.0-0')
             .pip_install('torch==2.5.1', 'torchvision==0.20.1', 'numpy==1.26.4', 'pillow==11.0.0', 'opencv-python-headless==4.10.0.84',
                          'huggingface_hub==0.36.0', 'safetensors==0.4.5', 'einops==0.8.0', 'trimesh==5.1.0', 'plyfile==1.1', 'pydantic==2.9.2',
                          'addict==2.4.0', 'omegaconf==2.3.0', 'hydra-core==1.3.2', 'imageio==2.36.0', 'tqdm==4.67.1', 'scipy==1.14.1',
                          'jaxtyping==0.2.36', 'timm==1.0.11', 'python-box==7.2.0', 'natsort==8.4.0', 'orjson==3.10.11', 'matplotlib==3.9.2',
                          'scikit-learn==1.5.2', 'termcolor==2.5.0')
             .pip_install('uniception==0.1.7', extra_options='--no-deps')
             .run_commands(f'git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /vendor/depth-anything-3 && git -C /vendor/depth-anything-3 checkout {DA3_CODE}')
             .add_local_python_source('argus'))


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
    from argus.pipeline.candidate_geometry_backend import DA3Runner
    from argus.providers.map_anything import MapAnythingAdapter
    start = time.monotonic(); timing = {}; root = Path(tempfile.mkdtemp(prefix='fair-ab-out-')); indir = Path(tempfile.mkdtemp(prefix='fair-ab-in-'))
    for bb, (repo, rev) in BACKBONES.items():
        t0 = time.monotonic(); wdir = Path(tempfile.mkdtemp(prefix='fair-ab-weights-')) / bb
        snapshot_download(repo, revision=rev, allow_patterns=['model.safetensors', 'config.json'], local_dir=wdir)
        runner = DA3Runner('/vendor/depth-anything-3', wdir, model_id=repo, device='cuda', process_res=518)
        t1 = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        for cell, variants in inputs.items():
            for variant, frames in variants.items():
                paths = []
                for name, data in sorted(frames.items()):
                    p = indir / cell / variant / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data); paths.append(str(p))
                out = root / f'{cell}-{bb}-{variant}' / 'geometry'
                MapAnythingAdapter(runner=runner).run(paths, out)
                meta = json.loads(json.dumps(runner.metadata, default=str)); meta['gpu'] = torch.cuda.get_device_name(0); meta['inputVariant'] = variant
                (out / 'candidate_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
                for junk in ('provider', 'map_anything_response.json', 'point_cloud.glb'):  # raw copies; frames/ keeps every array
                    p = out / junk; shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
        timing[bb] = dict(loadSeconds=t1 - t0, inferSeconds=time.monotonic() - t1, peakGiB=torch.cuda.max_memory_allocated() / 2 ** 30)
        del runner; torch.cuda.empty_cache()
    return dict(archive=pack(root), timing=timing, containerSeconds=time.monotonic() - start)




def analyse(job: dict) -> dict:
    import tempfile
    start = time.monotonic()
    import argus.checks.field_geometry as fa
    fa._check()
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(job['geometry']), mode='r:gz') as tar:
            tar.extractall(tmp, filter='data')
        out = analyse_one(job['cell'], job['backbone'], job['variant'], Path(tmp) / 'geometry', DATA)
    out['containerSeconds'] = time.monotonic() - start
    return out


def load_frames(cell: str, geom: Path, data: Path, variant: str):
    import numpy as np
    from PIL import Image
    import argus.checks.field_geometry as fa
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
    import argus.checks.clearance_geometry as cb
    import argus.checks.field_geometry as fa
    import argus.checks.geometry_measurements as ga
    import argus.checks.shape_core as wsc
    from argus.checks.estop_cylinder import fit
    from argus.pipeline.geometry import _ransac_floor_plane
    from argus.checks.reference_object_scale import joint_scale
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
def main(stage: str, cells: str = '090,030'):
    if stage != 'infer':
        raise ValueError(stage)
    selected = cells.split(',')
    for cell in selected:
        if (GEOM / f'{cell}-da3-base-padded').exists():
            raise ValueError(f'{cell} geometry exists; choose a fresh data root')
    t = time.monotonic(); r = infer.remote({c: frames_for(c) for c in selected})
    GEOM.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(r.pop('archive')), mode='r:gz') as tar:
        tar.extractall(GEOM, filter='data')
    ledger = dict(mode='ephemeral modal run', hardware='A100-80GB, 8 CPU, 32 GiB', functionSeconds=r['containerSeconds'], callSeconds=time.monotonic() - t,
                  estimateUsd=GPU_RATE * r['containerSeconds'], timing=r['timing'], rateSource='https://modal.com/pricing')
    (GEOM / f'spend-ledger-{cells}.json').write_text(json.dumps(ledger, indent=2) + '\n')
