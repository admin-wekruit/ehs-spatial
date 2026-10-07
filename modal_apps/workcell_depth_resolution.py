"""Paired MapAnything inputs on one ephemeral 2-A100 container; artifacts stay on Volume.

modal run modal_apps/workcell_depth_resolution.py --sources a.jpg,b.jpg,c.jpg,d.jpg --out RUN
Use --download only with >8 GiB free. No metrology claims: each branch has its own
native world; geometry and physical endpoints must be rederived before measuring.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import modal

if not modal.is_local():
    sys.path.insert(0, '/repo')
from modal_apps.workcell_photo_all import image, volumes, TORCH_HUB
from modal_apps.mapanything_app import CODE_REV, MODEL_ID

app = modal.App('workcell-depth-resolution')
RESULT_VOLUME = 'panoptes-fb-layers'
results_volume = modal.Volume.from_name(RESULT_VOLUME)
MOUNT = '/v/layers'
RESOURCE_RATE = 2 * .000694 + 16 * .0000131 + 80 * .00000222


@app.function(image=image, gpu='A100-80GB:2', cpu=16, memory=80*1024,
              volumes={**volumes, MOUNT: results_volume}, timeout=1200, retries=0, min_containers=0)
def experiment(images: list[bytes], run_id: str):
    started = time.monotonic()
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', run_id) or len(images) != 4:
        raise ValueError('A safe run id and exactly four source photos are required')
    root = Path(MOUNT) / 'workcell-depth-resolution' / run_id
    root.mkdir(parents=True, exist_ok=False)
    paths = []
    for i, payload in enumerate(images, 1):
        path = root / f'source-{i}.jpg'; path.write_bytes(payload); paths.append(str(path))
    env = os.environ.copy()
    env.update(HF_HOME='/v/map/huggingface', HF_HUB_OFFLINE='1', TORCH_HOME=TORCH_HUB,
               PYTHONPATH='/repo:/repo/scripts:/repo/modal_apps', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    summary = {'schemaVersion': 1, 'runId': run_id, 'modelId': MODEL_ID, 'codeRevision': CODE_REV,
        'artifactVolume': RESULT_VOLUME, 'cloudRunPath': str(root),
        'volumeRunPath': str(root.relative_to(MOUNT)), 'status': 'started',
        'inputSha256': [hashlib.sha256(payload).hexdigest() for payload in images],
        'control': 'same originals, cached model revision, seed and inference flags; custom gradient uses fixed raw-pixel footprint; upstream mask_edges remains raster-dependent, so this tests the resolution pipeline rather than pure neural output',
        'geometryStatus': 'new independent native worlds; no transferred old geometry or metric claims',
        'branches': {}}
    code_paths = [Path('/repo/scripts/workcell_map_worker.py'),
                  Path('/repo/modal_apps/mapanything_app.py'), Path('/repo/modal_apps/workcell_depth_resolution.py')]
    summary['implementationSha256'] = {str(p.relative_to('/repo')): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    try:
        # Freeze the single cached snapshot before either worker loads weights.
        resolved = subprocess.run(['/opt/mapanything/bin/python', '-c',
            'from huggingface_hub import snapshot_download; '
            f'print(snapshot_download({MODEL_ID!r}, local_files_only=True))'],
            env=env, capture_output=True, text=True, check=True, timeout=60).stdout.strip().splitlines()[-1]
        revision = Path(resolved).name
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Cannot identify the cached model snapshot commit')
        summary['modelRevision'] = revision

        def branch(name, gpu, size):
            target = root / name; target.mkdir()
            command = ['/opt/mapanything/bin/python', '/repo/scripts/workcell_map_worker.py',
                       str(target), *paths, '--model-revision', revision]
            if size:
                command += ['--fixed-size', *map(str, size)]
            branch_start = time.monotonic()
            record = {'gpu': gpu, 'requestedSizeWH': size,
                      'cloudFramesPath': str(target), 'volumeFramesPath': str(target.relative_to(MOUNT)),
                      'frameGlob': 'frame_*.json.gz', 'status': 'started'}
            with (target / 'worker.log').open('w') as log:
                proc = subprocess.Popen(command, env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                                        stdout=log, stderr=subprocess.STDOUT, text=True)
                try:
                    record['returncode'] = proc.wait(timeout=900)
                    record['status'] = 'completed' if proc.returncode == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    record['status'] = 'timeout'
                finally:
                    if proc.poll() is None:
                        proc.kill(); proc.wait(timeout=10)
                    record['wallSeconds'] = time.monotonic() - branch_start
            timing = target / 'geometry-timing.json'
            if timing.is_file():
                record['timing'] = json.loads(timing.read_text())
                expected = list(reversed(size or [392, 518]))
                if any(row['canonicalShapeHW'] != expected for row in record['timing']['frameSummaries']):
                    record.update(status='failed', error='Actual canonical raster differs from controlled target')
            else:
                record['logTail'] = (target / 'worker.log').read_text()[-3000:]
            print(json.dumps({'branch': name, 'status': record['status'], 'seconds': record['wallSeconds']}), flush=True)
            return name, record

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(branch, 'default-518', 0, None),
                       pool.submit(branch, 'double-1036', 1, [784, 1036])]
            for future in futures:
                name, record = future.result(); summary['branches'][name] = record
        summary['status'] = 'completed' if all(r['status'] == 'completed' for r in summary['branches'].values()) else 'failed'
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {str(exc)[-500:]}')
    finally:
        summary['containerWallSeconds'] = time.monotonic() - started
        summary['files'] = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
                             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in sorted(root.rglob('*')) if p.is_file()]
        (root / 'run.json').write_text(json.dumps(summary, indent=2) + '\n')
        results_volume.commit()
        # Returned compact record includes manifest hashing and Volume commit;
        # the committed record ends just before those bookkeeping operations.
        summary['containerWallSeconds'] = time.monotonic() - started
    return summary


def download_artifacts(summary, destination):
    # ponytail: keep large arrays on the existing Volume; download only when the
    # explicit disk reserve fits the complete manifest, with no background copier.
    size = sum(row['bytes'] for row in summary['files'])
    if shutil.disk_usage(destination).free <= 8 * 2**30 + size:
        raise ValueError('Downloading all arrays requires their byte count plus >8 GiB free')
    for row in summary['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe volume artifact path')
        target = destination / 'artifacts' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with target.open('xb') as stream:
            for block in results_volume.read_file(summary['volumeRunPath'] + '/' + relative.as_posix()):
                stream.write(block); digest.update(block)
        if target.stat().st_size != row['bytes'] or digest.hexdigest() != row['sha256']:
            raise ValueError('Downloaded artifact does not match the committed manifest')


@app.local_entrypoint()
def main(sources: str, out: str, download: bool = False):
    paths = [Path(p) for p in sources.split(',')]
    if len(paths) != 4 or len(set(p.resolve() for p in paths)) != 4 or not all(p.is_file() for p in paths):
        raise ValueError('Four distinct original JPEG files are required')
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    run_id = 'depth-' + uuid.uuid4().hex
    input_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    (destination / 'input-manifest.json').write_text(json.dumps({
        'runId': run_id, 'sources': [str(p) for p in paths], 'sha256': input_hashes,
        'artifactVolume': RESULT_VOLUME, 'volumeRunPath': f'workcell-depth-resolution/{run_id}'}, indent=2))
    ledger = {'mode': 'ephemeral modal run', 'hardware': '2 x A100-80GB; 16 CPU; 80 GiB',
        'status': 'started', 'actualBilledUsd': None, 'usdPerSecond': RESOURCE_RATE,
        'rateSource': 'https://modal.com/pricing', 'rateCheckedDate': '2026-10-01',
        'estimateBasis': 'reserved-resource list rate; call window includes scheduling; build excluded; not invoice'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2))
    start = time.monotonic()
    call_seconds = None
    try:
        summary = experiment.remote([p.read_bytes() for p in paths], run_id)
        call_seconds = time.monotonic() - start
        (destination / 'run.json').write_text(json.dumps(summary, indent=2) + '\n')
        ledger.update(status=summary['status'], functionSeconds=summary['containerWallSeconds'],
                      estimateUsd=RESOURCE_RATE * summary['containerWallSeconds'])
        if summary['inputSha256'] != input_hashes:
            raise ValueError('Cloud and caller source hashes disagree')
        if download:
            download_artifacts(summary, destination)
        if summary['status'] != 'completed':
            raise RuntimeError('Ablation failed; compact run.json preserves each branch status and cloud artifact path')
    except Exception as exc:
        ledger.update(status='failed', error=type(exc).__name__)
        raise
    finally:
        ledger['callSeconds'] = call_seconds if call_seconds is not None else time.monotonic() - start
        ledger['callWindowEstimateUsd'] = RESOURCE_RATE * ledger['callSeconds']
        (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'status': ledger['status'], 'runId': run_id,
                      'volumeRunPath': summary['volumeRunPath'], 'downloaded': download}))
