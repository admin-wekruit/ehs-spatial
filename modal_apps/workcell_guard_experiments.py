"""Bounded four-photo experiments in one ephemeral two-A100 allocation per call."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App('workcell-guard-experiments')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libgl1', 'libglib2.0-0', 'libgomp1')
         .pip_install('numpy==2.2.6', 'scipy==1.16.3', 'pillow', 'trimesh', 'pydantic>=2',
                      'shapely>=2', 'open3d==0.19.0', 'pylimap==2.0.0', 'pytlsd')
         .pip_install('torch==2.8.0', 'kornia==0.8.2')
         .env({'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}))
for directory in ('scripts', 'fast_report', 'ehs_spatial'):
    image = image.add_local_dir(REPO/directory, '/repo/'+directory, ignore=['**/__pycache__/**', '**/*.pyc'])


def unpack(payload, directory):
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        for m in archive.getmembers():
            if m.issym() or m.islnk() or Path(m.name).is_absolute() or '..' in Path(m.name).parts:
                raise ValueError('Unsafe experiment archive')
        archive.extractall(directory, filter='data')


@app.function(image=image, gpu='A100-80GB:2', cpu=16, memory=80*1024,
              timeout=1200, retries=0, min_containers=0)
def experiment(payload: bytes, mode: str):
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)/'input'; root.mkdir()
        out = Path(temp)/'output'; out.mkdir()
        unpack(payload, root)
        code_files = [Path('/repo/scripts')/name for name in ('workcell_guard_joint.py','check_workcell_guard_joint.py','workcell_guard_controls.py','workcell_guard_dense.py','workcell_guard_silhouette.py','workcell_photo_metrology.py','check_workcell_photo_metrology.py','workcell_photo_geometry.py','workcell_photo_calibration.py','workcell_photo_objects.py','workcell_photo_oneshot.py','check_workcell_camera_pixels.py') if (Path('/repo/scripts')/name).exists()]
        code_files += [Path('/repo/fast_report/x7.py'), Path('/repo/ehs_spatial/measurements.py')]
        (out/'implementation-manifest.json').write_text(json.dumps({str(p.relative_to('/repo')):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},indent=2))
        env = os.environ.copy(); env['PYTHONPATH'] = '/repo:/repo/scripts'
        sources = [str(root/f'source-{i}.jpg') for i in range(1,5)]
        records = {}
        def run(name, code, timeout):
            start = time.monotonic()
            command = ['python', '-c', code, str(root), str(out), *sources]
            try:
                p = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
                (out/f'{name}.log').write_text(p.stdout+'\n'+p.stderr)
                records[name] = {'returncode': p.returncode, 'seconds': time.monotonic()-start}
            except subprocess.TimeoutExpired:
                records[name] = {'status':'timeout', 'seconds':time.monotonic()-start}
            print(json.dumps({name:records[name]}), flush=True)
        prefix = 'from pathlib import Path; import sys; root=Path(sys.argv[1]); out=Path(sys.argv[2]); '
        if mode == 'controls':
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(run, 'alignment', prefix+'from scripts.workcell_guard_controls import alignment; alignment(root,out/"A1-similarity")', 240)
                b = pool.submit(run, 'colmap', prefix+'from scripts.workcell_guard_controls import colmap; colmap(root,out/"COLMAP",[Path(p) for p in sys.argv[3:]])', 360)
                a.result(); b.result()
            if records['colmap'].get('returncode') == 0:
                run('limap', prefix+'from scripts.workcell_guard_controls import limap_control; limap_control(out/"COLMAP",out/"LIMAP",root,[Path(p) for p in sys.argv[3:]])', 360)
        elif mode == 'joint':
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(run, 'joint', prefix+'from scripts.workcell_guard_joint import build; build(root,out/"original-cameras",sources=[Path(p) for p in sys.argv[3:]])', 600)]
                jobs.append(pool.submit(run, 'lk', prefix+'from scripts.workcell_guard_joint import build; build(root,out/"LK-original-cameras",sources=[Path(p) for p in sys.argv[3:]],feature_method="lk")',600))
                control = root/'control'
                if control.exists():
                    jobs.append(pool.submit(run, 'joint_colmap', prefix+'from scripts.workcell_guard_joint import build; build(root,out/"COLMAP-cameras",cameras=root/"control/cameras.json",tracks=root/"control/tracks.json",sources=[Path(p) for p in sys.argv[3:]])',600))
                    jobs.append(pool.submit(run, 'limap', prefix+'from scripts.workcell_guard_controls import limap_control; limap_control(root/"control",out/"LIMAP",root,[Path(p) for p in sys.argv[3:]])',360))
                    jobs.append(pool.submit(run, 'limap_guard', prefix+'from scripts.workcell_guard_controls import limap_control; limap_control(root/"control",out/"LIMAP-guard",root,[Path(p) for p in sys.argv[3:]],guard_only=True)',360))
                    jobs.append(pool.submit(run, 'lk_colmap', prefix+'from scripts.workcell_guard_joint import build; build(root,out/"LK-COLMAP-cameras",cameras=root/"control/cameras.json",sources=[Path(p) for p in sys.argv[3:]],feature_method="lk")',600))
                for job in jobs: job.result()
            if records.get('limap',{}).get('returncode') == 0:
                run('joint_limap', prefix+'from scripts.workcell_guard_joint import build; build(root,out/"LIMAP-cameras",cameras=out/"LIMAP/cameras.json",tracks=out/"LIMAP/tracks.json",sources=[Path(p) for p in sys.argv[3:]])',600)
        elif mode == 'dense':
            run('dense', prefix+'from scripts.workcell_guard_dense import run; run(root,out/"LoFTR-COLMAP-cameras",[Path(p) for p in sys.argv[3:]],root/"control/cameras.json")',900)
        elif mode == 'dense-replay':
            run('dense_replay', prefix+'from scripts.workcell_guard_joint import build; build(root,out/"LoFTR-COLMAP-cameras/joint",cameras=root/"control/cameras.json",tracks=root/"control/tracks.json",sources=[Path(p) for p in sys.argv[3:]])',600)
        elif mode == 'silhouette':
            run('silhouette', prefix+'from scripts.workcell_guard_silhouette import run; run(root,out/"A4-shared-silhouette",[Path(p) for p in sys.argv[3:]])',600)
        elif mode in ('metrology', 'metrology-square-pixels'):
            from concurrent.futures import ThreadPoolExecutor
            code = prefix+'import json; from scripts.workcell_photo_metrology import build; '
            args = 'sources=[Path(p) for p in sys.argv[3:]],reference=json.loads((root/"reference.json").read_text())'
            def square_pixels():
                run('square_pixel_cameras', prefix+'from scripts.workcell_guard_controls import colmap; colmap(root,out/"square-pixel-control",[Path(p) for p in sys.argv[3:]],square_pixels=True)',360)
                if records['square_pixel_cameras'].get('returncode') == 0:
                    run('metrology_square_pixels',code+'build(root,out/"square-pixel-cameras",'+args+',cameras=out/"square-pixel-control/cameras.json")',600)
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(run, 'metrology_original', code+'build(root,out/"original-cameras",'+args+')',600)]
                if mode == 'metrology-square-pixels':
                    jobs.append(pool.submit(square_pixels))
                elif (root/'control/cameras.json').is_file():
                    jobs.append(pool.submit(run, 'metrology_refined', code+'build(root,out/"refined-cameras",'+args+',cameras=root/"control/cameras.json")',600))
                for job in jobs: job.result()
        else:
            raise ValueError('Expected controls, joint, dense, dense-replay, silhouette, metrology or metrology-square-pixels')
        (out/'run.json').write_text(json.dumps({'records':records, 'containerWallSeconds':time.monotonic()-started}, indent=2))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.suffix in ('.json','.glb','.log','.txt','.bin','.ply','.npz','.png','.jpg'):
                    archive.add(path, arcname=str(path.relative_to(out)))
        return {'archive':buffer.getvalue(), 'containerWallSeconds':time.monotonic()-started}


@app.local_entrypoint()
def main(baseline: str, out: str, sources: str, mode: str = 'controls', control: str = '', alignment: str = '', measurements: str = ''):
    root = Path(baseline); destination = Path(out)
    if destination.exists(): raise ValueError('Use a fresh experiment output directory')
    reference = None
    if mode in ('metrology', 'metrology-square-pixels'):
        if not measurements: raise ValueError('Metrology requires --measurements with known reference dimensions')
        from scripts.workcell_photo_calibration import load_measurements
        config = load_measurements(measurements)
        # Only reference dimensions enter the solver; check distances stay on the caller.
        reference = json.dumps({'schemaVersion':1,'reference':config['reference']}).encode()
    source_paths = [Path(p) for p in sources.split(',')]
    if len(source_paths) != 4 or not all(p.is_file() for p in source_paths): raise ValueError('Four source photos required')
    files = [*root.glob('frame_*.json.gz'), *root.glob('photo-*.png')]
    files += [root/name for name in ('geometry.json','sam3.json','objects.json','guard-input.npz',
              'guard-placement.json','guard-partition.json','guard-multi.glb','guard-left.glb','guard-center.glb','guard-right.glb')]
    if not all(p.is_file() for p in files): raise ValueError('Incomplete frozen baseline')
    buffer = io.BytesIO(); hashes = {}
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path, name in [(p,p.name) for p in files]+[(p,f'source-{i}.jpg') for i,p in enumerate(source_paths,1)]:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest(); archive.add(path, arcname=name)
        if reference is not None:
            item = tarfile.TarInfo('reference.json'); item.size = len(reference)
            hashes[item.name] = hashlib.sha256(reference).hexdigest()
            archive.addfile(item, io.BytesIO(reference))
        if control:
            directory = Path(control)
            for p in [directory/'cameras.json',directory/'tracks.json',*sorted((directory/'model').glob('*.bin'))]:
                name = 'control/'+str(p.relative_to(directory))
                hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest(); archive.add(p,arcname=name)
        if alignment:
            for filename in ('guard-left.glb','guard-right.glb','results.json'):
                p=Path(alignment)/filename; name='a1/'+filename
                hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest(); archive.add(p,arcname=name)
    destination.mkdir(parents=True)
    control_result = Path(control)/'results.json' if control else None
    control_model = json.loads(control_result.read_text()).get('cameraModel') if control_result and control_result.is_file() else None
    (destination/'input-manifest.json').write_text(json.dumps({'baseline':str(root),'mode':mode,
        'control':control or None,'controlCameraModel':control_model,'sha256':hashes},indent=2))
    start = time.monotonic()
    rate = 2*.000694+16*.0000131+80*.00000222
    ledger = {'mode':'ephemeral modal run','hardware':'2 x A100-80GB; 16 CPU; 80 GiB',
              'actualBilledUsd':None, 'status':'started', 'usdPerSecond':rate,
              'rateSource':'https://modal.com/pricing', 'rateCheckedDate':'2026-10-01',
              'estimateBasis':'reserved-resource list rate; call window includes scheduling; build time excluded; not invoice'}
    try:
        result = experiment.remote(buffer.getvalue(), mode)
        unpack(result['archive'], destination)
        elapsed = time.monotonic()-start
        ledger.update(status='completed', functionSeconds=result['containerWallSeconds'],callSeconds=elapsed,
                      estimateUsd=rate*result['containerWallSeconds'],callWindowEstimateUsd=rate*elapsed,
                      usdPerSecond=rate, rateSource='https://modal.com/pricing', rateCheckedDate='2026-10-01',
                      estimateBasis='reserved-resource list rate; call window includes scheduling; build time excluded; not invoice')
        records = json.loads((destination/'run.json').read_text())['records']
        if any(record.get('returncode') != 0 for record in records.values()):
            raise RuntimeError('Experiment subprocess failed; inspect saved run.json and logs')
    except Exception as exc:
        ledger.update(status='failed', callSeconds=time.monotonic()-start, error=type(exc).__name__)
        raise
    finally:
        ledger['callSeconds'] = time.monotonic()-start
        ledger['callWindowEstimateUsd'] = rate*ledger['callSeconds']
        (destination/'spend-ledger.json').write_text(json.dumps(ledger,indent=2))
    print(json.dumps(ledger))
