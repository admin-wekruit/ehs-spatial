"""Geometry backbone A/B (2026-10-06): licence-clean candidates in the fair harness of geometry-licence-ab-fair-2026-10-05.

Ephemeral Modal functions only (nothing deployed), every one with an explicit timeout, retries=0, min_containers=0.
  modal run backbone_ab_modal.py --stage access     can the Modal 'huggingface' secret read facebook/VGGT-1B-Commercial? (status only)
  modal run backbone_ab_modal.py --stage classic    L4: MoGe-3 on every frozen frame -> our K; ALIKED + LightGlue poses with our K;
                                                    MoGe-3 depth aligned to the triangulated points -> GEOM/CELL-classic-padded
  python mvs_route.py BASE [CERT ...]               local: the completed MVS route -> GEOM/CELL-mvs-BASE-padded
  python backbone_ab_modal.py analyse [names]       backbones.check_geometry on every new geometry, then the UNCHANGED fair_ab_modal.analyse
                                                    (its own app, ephemeral) -> OUT/CELL-BACKBONE-padded.json
  python backbone_ab_modal.py export BACKBONE CELL  run directory in the lucida-replica-01 layout for the downstream completion A/B
VGGT-1B-Commercial and map-anything-apache (+ BA) run in licence-clean-stack-2026-10-06/geometry (modal_apps/geometry_clean_ab.py).
Every backbone gets the frozen padded 518 canonical frames of the run, and emits the frames/<id>/ files of MapAnythingAdapter.
"""
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import modal

NOTE = Path(__file__).resolve().parent
FAIR = NOTE.parent / 'geometry-licence-ab-fair-2026-10-05'
SERV = Path(os.environ.get('PANOPTES_SERVING', '/Users/adam/Desktop/panoptes-public/panoptes-serving'))  # = fair_ab_modal's paths (it is imported locally only:
SCR = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))  # not in the images)
RUNS = Path(os.environ.get('PANOPTES_RUNS', '/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation'))
GEOM, OUT = SCR / 'checks/bbab-geom', SCR / 'checks/bbab-analyse'
L4_RATE = .000222 + 4 * .0000131 + 16 * .00000222  # L4 + 4 CPU + 16 GiB list rate (USD/s); not an invoice
VGGT = ('facebook/VGGT-1B-Commercial', 'ebb29a532abe92960eeb6903a5530f16990ef4ab')
MOGE_CODE, LIGHTGLUE_CODE = '74fbce054ebed49800de42d0ad0e83495065719a', 'eb42fee2d71449efb0aa5c10549752b5d75384d8'

app = modal.App('geometry-backbone-ab')
moge_image = (modal.Image.debian_slim(python_version='3.11')  # modal_apps/moge3_app.py's recipe, pinned; + LightGlue (ALIKED) + the serving seam
              .apt_install('git', 'build-essential', 'libgl1', 'libglib2.0-0')
              .pip_install('torch==2.8.0', 'torchvision==0.23.0')
              .pip_install(f'git+https://github.com/microsoft/MoGe.git@{MOGE_CODE}', f'git+https://github.com/cvg/LightGlue.git@{LIGHTGLUE_CODE}',
                           'huggingface_hub', 'opencv-python-headless', 'trimesh', 'pydantic', 'scipy')
              .env({'HF_HOME': '/cache/huggingface'})
              .add_local_dir(SERV / 'ehs_spatial', '/serving/ehs_spatial', ignore=['__pycache__'])
              .add_local_file(SERV / 'scripts/candidate_geometry_backend.py', '/serving/scripts/candidate_geometry_backend.py')
              .add_local_file(NOTE / 'backbones.py', '/serving/scripts/backbones.py'))
hf_secret = modal.Secret.from_name('huggingface')


def pack(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        tar.add(root, arcname='.')
    return buf.getvalue()


def harness():
    """The fair harness module (local side only): frames_for, geometry_tar, analyse + its app, CELLS, rates."""
    sys.path[:0] = [str(NOTE), str(FAIR)]
    import fair_ab_modal
    return fair_ab_modal


@app.function(image=modal.Image.debian_slim(python_version='3.11').pip_install('huggingface_hub==0.36.0'), cpu=1, memory=1024,
              timeout=300, retries=0, min_containers=0, secrets=[hf_secret])
def access() -> dict:
    """Status only: never returns the credential, error text or URLs (they may carry it)."""
    import os
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    present = any(os.environ.get(k) for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGINGFACE_HUB_TOKEN'))
    out = dict(repo=VGGT[0], revision=VGGT[1], credentialPresent=present, checks={})
    for name, act in (('whoami', lambda: HfApi().whoami()['name'] and None),
                      ('config.json', lambda: get_hf_file_metadata(hf_hub_url(VGGT[0], 'config.json', revision=VGGT[1])).size),
                      ('model.safetensors', lambda: get_hf_file_metadata(hf_hub_url(VGGT[0], 'model.safetensors', revision=VGGT[1])).size)):
        try:
            out['checks'][name] = dict(status='accessible', sizeBytes=act())
        except Exception as error:  # noqa: BLE001 - classified, never echoed
            r = getattr(error, 'response', None)
            out['checks'][name] = dict(status='refused', type=type(error).__name__, httpStatus=getattr(r, 'status_code', None),
                                       errorCode=(r.headers.get('x-error-code') if r is not None else None))
    out['granted'] = out['checks']['model.safetensors']['status'] == 'accessible'
    return out


@app.function(image=moge_image, gpu='L4', cpu=4, memory=16 * 1024, timeout=1800, retries=0, min_containers=0,
              volumes={'/cache': modal.Volume.from_name('moge3-hf-cache')})
def classic(inputs: dict) -> dict:
    """inputs = {cell: {name: png}} (frozen padded frames). MoGe-3 -> our K; classic poses + aligned MoGe depth -> geometry."""
    import numpy as np
    import torch
    sys.path[:0] = ['/serving', '/serving/scripts']
    import backbones as bb
    from ehs_spatial.providers.map_anything import MapAnythingAdapter
    start = time.monotonic(); root = Path('/tmp/out'); report = {}
    moge = bb.MoGe3('cuda'); t_load = time.monotonic() - start; focal = None
    for cell, frames in sorted(inputs.items(), key=lambda kv: -len(kv[1])):  # the three-view cell first: it sets the device focal
        paths = []
        for name, data in sorted(frames.items()):
            p = Path('/tmp/in') / cell / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data); paths.append(str(p))
        runner = bb.ClassicRunner(moge, device='cuda', focal=focal)
        out = root / f'{cell}-classic-padded' / 'geometry'
        MapAnythingAdapter(runner=runner).run(paths, out)
        np.savez(out.parent / 'matches.npz', **{f'kps{i}': k for i, k in enumerate(runner.kps)},
                 **{f'm{i}{j}': m for (i, j), m in runner.matches.items()})  # for offline checks; not part of the contract
        bb.finish(out, runner.metadata | dict(gpu=torch.cuda.get_device_name(0)))
        report[cell] = {k: runner.metadata[k] for k in ('ourK', 'ourKSource', 'focalSweep', 'mogeMedianK', 'mogeFocalPx')}
        focal = runner.metadata['ourK'][0][0]
    return dict(archive=pack(root), report=report, loadSeconds=t_load, containerSeconds=time.monotonic() - start)


def padded_inputs(cells):
    return {c: harness().frames_for(c)['padded'] for c in cells}


def unpack(archive: bytes):
    GEOM.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        names = {Path(m.name).parts[1] for m in tar.getmembers() if len(Path(m.name).parts) > 1}
        clash = [n for n in names if (GEOM / n).exists()]
        if clash:
            raise ValueError(f'{clash} exist in {GEOM}; choose fresh names')
        tar.extractall(GEOM, filter='data')


def ledger(name, rate, hardware, r, t):
    row = dict(stage=name, mode='ephemeral modal run', hardware=hardware, functionSeconds=r['containerSeconds'], callSeconds=time.monotonic() - t,
               estimateUsd=rate * r['containerSeconds'], rateSource='https://modal.com/pricing', **{k: v for k, v in r.items() if k in ('timing', 'report', 'loadSeconds')})
    (GEOM / f'spend-{name}-{int(time.time())}.json').write_text(json.dumps(row, indent=2) + '\n'); print(json.dumps(row)[:2000])


@app.local_entrypoint()
def main(stage: str, cells: str = '090,030'):
    cells = cells.split(',')
    if stage == 'access':
        r = access.remote(); GEOM.mkdir(parents=True, exist_ok=True)
        (GEOM / 'vggt-access.json').write_text(json.dumps(r, indent=2) + '\n'); print(json.dumps(r))
        return
    t = time.monotonic()
    if stage == 'classic':
        r = classic.remote(padded_inputs(cells)); unpack(r.pop('archive')); ledger('classic', L4_RATE, 'L4, 4 CPU, 16 GiB', r, t)
        (GEOM / 'our-k.json').write_text(json.dumps(r['report'], indent=2) + '\n')
        return
    raise ValueError(stage)


def analyse_local(names):
    """The fair harness's analyse function and app, unchanged; the contract check runs on each geometry first."""
    import backbones as bb
    fair = harness(); jobs = []
    for d in sorted(GEOM.glob('*-padded')):
        cell, backbone = d.name.split('-', 1)[0], d.name.split('-', 1)[1].rsplit('-', 1)[0]
        if names and d.name not in names:
            continue
        bb.check_geometry(d / 'geometry', RUNS / fair.CELLS[cell]['run'])
        jobs.append(dict(cell=cell, backbone=backbone, variant='padded', geometry=fair.geometry_tar(d / 'geometry')))
    print('contract passed:', [f"{j['cell']}-{j['backbone']}" for j in jobs])
    OUT.mkdir(parents=True, exist_ok=True); t = time.monotonic(); total = 0.
    with modal.enable_output(), fair.app.run():
        for job, r in zip(jobs, fair.analyse.map(jobs, order_outputs=True, return_exceptions=True)):
            name = f"{job['cell']}-{job['backbone']}-{job['variant']}"
            if isinstance(r, Exception):
                print(name, 'FAILED', repr(r)[:800]); continue
            (OUT / f'{name}.json').write_text(json.dumps(r, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
            total += r['containerSeconds']; print(name, f"{r['containerSeconds']:.0f}s")
    row = dict(stage='analyse', mode='ephemeral modal run (fair_ab_modal app, unchanged)', hardware='8 CPU, 16 GiB per job', jobs=len(jobs),
               functionSeconds=total, callSeconds=time.monotonic() - t, estimateUsd=fair.CPU_RATE * total, rateSource='https://modal.com/pricing')
    (OUT / f'spend-ledger-{int(time.time())}.json').write_text(json.dumps(row, indent=2) + '\n'); print(json.dumps(row))


def export_run(backbone: str, cell: str = '090'):
    """The run directory the downstream completion A/B reads (lucida-replica-01 layout), with this backbone's geometry:
    geometry/frames/<id>/ (+ content_valid_mask.npy by prepare_capture_evidence.RULE), evidence/objects/<obj>/<id>/points.npy +
    colors.npy re-derived (mask & content, as prepare_lucida_evidence), objects.json centroids (median), floor.json = the fair
    A/B primary floor (maxInlier) on the frozen floor masks + this backbone's e-stop scale. Image-space files (input photos,
    masks, rgba crops) are symlinked read-only; canonical frames copied. Nothing in the source run is written."""
    import hashlib
    import os
    import shutil
    import numpy as np
    from PIL import Image
    import backbones as bb
    fair = harness(); src = RUNS / fair.CELLS[cell]['run']; geom = GEOM / f'{cell}-{backbone}-padded/geometry'
    dst = SCR / f'checks/bbab-export-{cell}-{backbone}'
    if dst.exists():
        raise ValueError(f'{dst} exists')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    man = json.loads((src / 'manifest.json').read_text()); bb.check_geometry(geom, src)
    (dst / 'evidence').mkdir(parents=True); os.symlink(src / 'input', dst / 'input')
    shutil.copytree(src / 'evidence/canonical', dst / 'evidence/canonical')
    P, content, rgb, records = {}, {}, {}, []
    for f in man['frames']:
        fid = f['frame_id']; g = dst / 'geometry/frames' / fid; g.mkdir(parents=True)
        for name in ('pts3d.npy', 'conf.npy', 'valid_mask.npy', 'intrinsics.npy', 'camera_to_world.npy'):
            shutil.copy(geom / 'frames' / fid / name, g / name)
        shutil.copy(src / f['canonical'], g / 'canonical.png')  # the frozen bytes (check_geometry asserted the same pixels)
        P[fid] = np.load(g / 'pts3d.npy'); conf = np.load(g / 'conf.npy'); valid = np.load(g / 'valid_mask.npy').astype(bool)
        content[fid] = valid & np.load(src / f['alpha']) & np.isfinite(P[fid]).all(-1) & (np.linalg.norm(P[fid], axis=-1) > 1e-6) & (conf >= .1)
        np.save(g / 'content_valid_mask.npy', content[fid]); rgb[fid] = np.asarray(Image.open(g / 'canonical.png').convert('RGB'))
        records.append(dict(frame_id=fid, valid_content_points=int(content[fid].sum()), canonical_pixels=int(content[fid].size),
                            files={p.name: digest(p) for p in sorted(g.iterdir())}))
    shutil.copy(geom / 'candidate_manifest.json', dst / 'geometry/candidate_manifest.json')

    def rederive(v):
        d = dst / Path(v['canonical_mask_path']).parent; d.mkdir(parents=True, exist_ok=True)
        for p in (v['canonical_mask_path'], v['mask_path'], v['rgba_path']):
            os.symlink(src / p, dst / p)
        m = np.load(src / v['canonical_mask_path']).astype(bool) & content[v['frame_id']]
        np.save(dst / v['points_path'], P[v['frame_id']][m]); np.save(dst / v['colors_path'], rgb[v['frame_id']][m])
        v.update(partial_point_count=int(m.sum()), centroid_native=np.median(P[v['frame_id']][m], axis=0).tolist() if m.any() else None,
                 sha256={Path(p).name: digest(dst / p) for p in (v['points_path'], v['colors_path'])})
        return P[v['frame_id']][m], rgb[v['frame_id']][m]
    objs = json.loads((src / 'evidence/objects.json').read_text())
    for o in objs['objects']:
        for v in o['views']:
            rederive(v)
        c = [np.array(v['centroid_native']) for v in o['views'] if v['centroid_native'] is not None]
        o['physical_identity']['pairwise_visible_centroid_distances_native'] = [[float(np.linalg.norm(a - b)) for b in c] for a in c]
        o['physical_identity'].pop('two_view_centroid_distance_native', None)
    objs['coordinate_system'] = f'{backbone} native OpenCV world (geometry-backbone-ab export; not the published Pi3X world)'
    (dst / 'evidence/objects.json').write_text(json.dumps(objs, indent=2) + '\n')
    fl = json.loads((src / 'evidence/floor.json').read_text()); pts, cols = zip(*[rederive(v) for v in fl['views']])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    np.save(dst / 'evidence/floor_points.npy', pts); np.save(dst / 'evidence/floor_colors.npy', cols)
    a = json.loads((OUT / f'{cell}-{backbone}-padded.json').read_text())['floors']['maxInlier']; n, d = np.array(a['normal']), a['offset']
    res = np.abs(pts @ n + d)
    fl.update(plane_native=[*n.tolist(), d], up_native=n.tolist(), coordinate_system=objs['coordinate_system'],
              fit='geometry-licence-ab-fair maxInlier (primary): largest-consensus plane, 20 deg gate, cameras above, two SVD refits',
              distance_threshold_native=None, total_observed_points=len(pts), all_point_residual_median_native=float(np.median(res)),
              all_point_residual_p95_native=float(np.quantile(res, .95)), inlier_points=None, inlier_residual_p95_native=None,
              camera_signed_heights_native=a['cameraHeightsNative'], estopNativeToMeters=a['estop']['nativeToMeters'],
              estopMaxDeviation=a['estop']['maxDeviation'], estopScaleSource='this backbone\'s own e-stop (fair harness); not a published scale')
    (dst / 'evidence/floor.json').write_text(json.dumps(fl, indent=2) + '\n')
    man['geometry'] = dict(status='complete', path='geometry', model=backbone, input_frames=[f['frame_id'] for f in man['frames']],
                           coordinate_system=objs['coordinate_system'], metric_scale_known=False, frames=records,
                           content_valid_rule='native valid & declared alpha & finite & nonzero points & conf>=0.1',
                           candidate=json.loads((geom / 'candidate_manifest.json').read_text()).get('model_id'))
    man['evidence'] = dict(objects='evidence/objects.json', objects_sha256=digest(dst / 'evidence/objects.json'), floor='evidence/floor.json',
                           floor_sha256=digest(dst / 'evidence/floor.json'), exportedFrom=str(src), exportedBy=str(Path(__file__).resolve()))
    man['experiment'] = dst.name
    (dst / 'manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    bb.check_geometry(dst / 'geometry', src)
    print(dst, {o['object_id']: [v['partial_point_count'] for v in o['views']] for o in objs['objects']}, 'floor points', len(pts))


if __name__ == '__main__' and sys.argv[1:2] == ['analyse']:
    analyse_local(sys.argv[2:])
if __name__ == '__main__' and sys.argv[1:2] == ['export']:
    export_run(*sys.argv[2:])
