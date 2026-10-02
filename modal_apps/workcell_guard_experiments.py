"""Bounded four-photo experiments in one ephemeral two-A100 allocation per call."""
import hashlib
import io
import json
import os
import signal
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
              volumes={'/v/layers': modal.Volume.from_name('panoptes-fb-layers')},
              timeout=2100, retries=0, min_containers=0)
def experiment(payload: bytes, mode: str, joint_max_nfev: int = 100):
    # Direct imports and child processes use the same mounted source tree.
    sys.path[:0] = ['/repo', '/repo/scripts']
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)/'input'; root.mkdir()
        out = Path(temp)/'output'; out.mkdir()
        unpack(payload, root)
        code_files = [Path('/repo/scripts')/name for name in ('workcell_guard_joint.py','check_workcell_guard_joint.py','workcell_guard_controls.py','workcell_guard_dense.py','workcell_guard_silhouette.py','workcell_photo_metrology.py','check_workcell_photo_metrology.py','workcell_photo_geometry.py','workcell_photo_calibration.py','workcell_photo_objects.py','workcell_photo_oneshot.py','check_workcell_camera_pixels.py') if (Path('/repo/scripts')/name).exists()]
        code_files += [Path('/repo/fast_report/x7.py'), Path('/repo/ehs_spatial/measurements.py')]
        if mode in ('metrology-joint-button', 'metrology-depth', 'real2sim', 'real2sim-details'):
            code_files += [Path('/repo/scripts')/name for name in ('workcell_button_bundle.py', 'check_workcell_button_bundle.py')]
        if mode in ('real2sim', 'real2sim-details', 'physical-bottoms'):
            code_files += [Path('/repo/scripts')/name for name in ('workcell_real2sim_experiment.py', 'workcell_post_faces.py', 'workcell_photo_texture.py')]
        if mode == 'physical-bottoms':
            code_files += [Path('/repo/scripts')/name for name in ('workcell_fence_bottom.py', 'workcell_physical_bottoms.py',
                'workcell_bottom_fit.py', 'workcell_bottom_models.py')]
        if mode == 'metrology-depth':
            code_files += [Path('/repo/scripts')/name for name in ('workcell_depth_metrology.py', 'check_workcell_depth_metrology.py')]
        (out/'implementation-manifest.json').write_text(json.dumps({str(p.relative_to('/repo')):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},indent=2))
        env = os.environ.copy(); env['PYTHONPATH'] = '/repo:/repo/scripts'
        sources = [str(root/f'source-{i}.jpg') for i in range(1,5)]
        records = {}
        def run(name, code, timeout):
            start = time.monotonic()
            command = ['python', '-c', code, str(root), str(out), *sources]
            try:
                p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, env=env, start_new_session=True)
                stdout, stderr = p.communicate(timeout=timeout)
                (out/f'{name}.log').write_text(stdout+'\n'+stderr)
                records[name] = {'returncode': p.returncode, 'seconds': time.monotonic()-start}
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = p.communicate()
                (out/f'{name}.log').write_text(stdout+'\n'+stderr)
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
        elif mode == 'physical-bottoms':
            run('physical_bottoms', prefix+'from scripts.workcell_physical_bottoms import build; '
                'build(root, out/"physical-bottoms", [Path(p) for p in sys.argv[3:]])', 600)
        elif mode in ('real2sim', 'real2sim-details'):
            run('real2sim', prefix+'from scripts.workcell_real2sim_experiment import run; '
                f'run(root,out/"real2sim",[Path(p) for p in sys.argv[3:]],max_nfev={joint_max_nfev},details_only={mode == "real2sim-details"})',
                300 if mode == 'real2sim-details' else 1500)
        elif mode == 'metrology-depth':
            from concurrent.futures import ThreadPoolExecutor
            from scripts.workcell_depth_metrology import depth_run_path
            depth_path = depth_run_path('/v/layers', json.loads((root/'depth-run.json').read_text())['volumeRunPath'])
            summary = json.loads((depth_path/'run.json').read_text())
            if summary['inputSha256'] != [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources]:
                raise ValueError('Depth ablation and metrology source photos disagree')
            jobs = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                for name in ('default-518', 'double-1036'):
                    if summary['branches'].get(name, {}).get('status') != 'completed':
                        records[name] = {'status': 'depth_unavailable', 'reason': 'Source depth branch did not complete'}
                        continue
                    fresh = out/name/'fresh-geometry'; fresh.mkdir(parents=True)
                    import shutil
                    for p in sorted((depth_path/name).glob('*')):
                        if p.name.startswith(('frame_', 'photo-')) and p.is_file():
                            shutil.copyfile(p, fresh/p.name)
                    code = prefix+'import json; from scripts.workcell_depth_metrology import run; '
                    code += f'run(root,out/{name!r}/"fresh-geometry",out/{name!r}/"measurements",'
                    code += f'sources=[Path(p) for p in sys.argv[3:]],reference=json.loads((root/"reference.json").read_text()),joint_max_nfev={joint_max_nfev})'
                    jobs.append(pool.submit(run, name, code, 1050))
                for job in jobs: job.result()
            run('cached_original', prefix+'import json; from scripts.workcell_photo_metrology import build; '
                'build(root,out/"cached-original-cameras",sources=[Path(p) for p in sys.argv[3:]],'
                'reference=json.loads((root/"reference.json").read_text()))',180)
            if (root/'control/cameras.json').is_file():
                run('cached_joint_button', prefix+'import json; from scripts.workcell_button_bundle import build; '
                    'build(root,out/"cached-joint-button",sources=[Path(p) for p in sys.argv[3:]],'
                    'reference=json.loads((root/"reference.json").read_text()),'
                    f'cameras=root/"control/cameras.json",tracks=root/"control/tracks.json",max_nfev={joint_max_nfev})',500)
                if (out/'cached-joint-button/cameras.json').is_file():
                    run('cached_joint_metrology',prefix+'import json; from scripts.workcell_photo_metrology import build; '
                        'build(root,out/"cached-joint-cameras",sources=[Path(p) for p in sys.argv[3:]],'
                        'reference=json.loads((root/"reference.json").read_text()),'
                        'cameras=out/"cached-joint-button/cameras.json",joint_reference=out/"cached-joint-button/joint-reference.json")',180)
        elif mode == 'metrology-joint-button':
            run('joint_button', prefix+'import json; from scripts.workcell_button_bundle import build; '
                'build(root,out/"joint-button",sources=[Path(p) for p in sys.argv[3:]],'
                'reference=json.loads((root/"reference.json").read_text()),'
                f'cameras=root/"control/cameras.json",tracks=root/"control/tracks.json",max_nfev={joint_max_nfev})',950)
            joint_dir = out/'joint-button'
            if records['joint_button'].get('returncode') == 0 and (joint_dir/'cameras.json').is_file():
                run('metrology_joint_button',prefix+'import json; from scripts.workcell_photo_metrology import build; '
                    'build(root,out/"joint-cameras",sources=[Path(p) for p in sys.argv[3:]],'
                    'reference=json.loads((root/"reference.json").read_text()),'
                    'cameras=out/"joint-button/cameras.json",joint_reference=out/"joint-button/joint-reference.json")',180)
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
            raise ValueError('Unknown experiment mode')
        (out/'run.json').write_text(json.dumps({'records':records, 'containerWallSeconds':time.monotonic()-started}, indent=2))
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.suffix in ('.json','.glb','.log','.txt','.bin','.ply','.npz','.png','.jpg'):
                    archive.add(path, arcname=str(path.relative_to(out)))
        return {'archive':buffer.getvalue(), 'containerWallSeconds':time.monotonic()-started}


@app.local_entrypoint()
def main(baseline: str, out: str, sources: str, mode: str = 'controls', control: str = '', alignment: str = '', measurements: str = '', depth_run: str = '', joint_max_nfev: int = 100):
    root = Path(baseline); destination = Path(out)
    if not 1 <= joint_max_nfev <= 400:
        raise ValueError('Joint iteration cap must be between 1 and 400')
    if destination.exists(): raise ValueError('Use a fresh experiment output directory')
    reference = None
    if mode in ('metrology', 'metrology-square-pixels', 'metrology-joint-button', 'metrology-depth', 'real2sim', 'real2sim-details'):
        if not measurements: raise ValueError('Metrology requires --measurements with known reference dimensions')
        from scripts.workcell_photo_calibration import load_measurements
        config = load_measurements(measurements)
        # Only reference dimensions enter the solver; check distances stay on the caller.
        reference = json.dumps({'schemaVersion':1,'reference':config['reference']}).encode()
    if mode == 'metrology-joint-button' and (not control or not all((Path(control)/name).is_file() for name in ('cameras.json','tracks.json'))):
        raise ValueError('Joint button fitting requires saved source camera and track files')
    if mode == 'metrology-depth':
        from scripts.workcell_depth_metrology import depth_run_path
        summary = json.loads(Path(depth_run).read_text())
        relative = Path(summary['volumeRunPath'])
        depth_run_path('/v/layers', relative)
    source_paths = [Path(p) for p in sources.split(',')]
    if len(source_paths) != 4 or not all(p.is_file() for p in source_paths): raise ValueError('Four source photos required')
    files = [*root.glob('frame_*.json.gz'), *root.glob('photo-*.png')]
    files += [root/name for name in ('geometry.json','sam3.json','objects.json')]
    if mode == 'physical-bottoms':
        files += [root/name for name in ('posts.glb','fence-fitted.glb','physical-clearances.json')]
    else:
        files += [root/name for name in ('guard-input.npz','guard-placement.json','guard-partition.json',
                  'guard-multi.glb','guard-left.glb','guard-center.glb','guard-right.glb')]
    if mode in ('real2sim', 'real2sim-details'):
        files += [root/name for name in ('posts.glb', 'posts-source.json', 'physical-clearances.json',
                                         'fence-fitted.glb', 'floor-fitted.glb', 'structural-result.json')]
    if not all(p.is_file() for p in files): raise ValueError('Incomplete frozen baseline')
    buffer = io.BytesIO(); hashes = {}
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path, name in [(p,p.name) for p in files]+[(p,f'source-{i}.jpg') for i,p in enumerate(source_paths,1)]:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest(); archive.add(path, arcname=name)
        if reference is not None:
            item = tarfile.TarInfo('reference.json'); item.size = len(reference)
            hashes[item.name] = hashlib.sha256(reference).hexdigest()
            archive.addfile(item, io.BytesIO(reference))
        if mode == 'metrology-depth':
            payload = json.dumps({'volumeRunPath': relative.as_posix()}).encode()
            item = tarfile.TarInfo('depth-run.json'); item.size = len(payload)
            hashes[item.name] = hashlib.sha256(payload).hexdigest()
            archive.addfile(item, io.BytesIO(payload))
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
    (destination/'input-manifest.json').write_text(json.dumps({'baseline':str(root),'mode':mode,'jointMaxNfev':joint_max_nfev,
        'control':control or None,'controlCameraModel':control_model,'sha256':hashes},indent=2))
    start = time.monotonic()
    rate = 2*.000694+16*.0000131+80*.00000222
    ledger = {'mode':'ephemeral modal run','hardware':'2 x A100-80GB; 16 CPU; 80 GiB',
              'actualBilledUsd':None, 'status':'started', 'usdPerSecond':rate,
              'rateSource':'https://modal.com/pricing', 'rateCheckedDate':'2026-10-02',
              'estimateBasis':'reserved-resource list rate; call window includes scheduling; build time excluded; not invoice'}
    try:
        result = experiment.remote(buffer.getvalue(), mode, joint_max_nfev)
        unpack(result['archive'], destination)
        elapsed = time.monotonic()-start
        ledger.update(status='completed', functionSeconds=result['containerWallSeconds'],callSeconds=elapsed,
                      estimateUsd=rate*result['containerWallSeconds'],callWindowEstimateUsd=rate*elapsed,
                      usdPerSecond=rate, rateSource='https://modal.com/pricing', rateCheckedDate='2026-10-02',
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
