"""Ephemeral 2 x A100 RecGen for small catalog objects of one or more scenes; inputs built locally, placement in the container.

  modal run modal_apps/workcell_recgen_objects.py --run RUN --out OUT --sources a.jpg,b.jpg,c.jpg,d.jpg \
      --scenes "cell-090=3,4:signal-light-2,signal-light-4;cell-030=1,2:signal-light-2,signal-light-4" [--control-canonical]

Local: scripts/workcell_recgen_objects.build_inputs per scene (npz bytes in memory; nothing large on local disk).
Container (the oneshot's image; one 2 x A100-80GB allocation, explicit timeout, retries 0, min_containers 0): one
scripts/workcell_recgen_worker.py process per GPU (objects dealt across both, one RecGen load per GPU), then
workcell_recgen_objects.place per object on the container's CPUs. Back home: {kind}.glb and {kind}.json per object,
the worker reports, review overlays ({kind}-overlay.jpg, CPU raster), outcome.json and spend-ledger.json (every call,
failed ones included; list-rate estimate, not an invoice). --control-canonical adds the same objects from the
canonical frames (no original-photo resampling) as labelled controls in the same call.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time

import modal

if not modal.is_local():
    sys.path.insert(0, '/repo')
from modal_apps.workcell_photo_all import image, volumes, _start, _finish

app = modal.App('workcell-recgen-objects')
USD_PER_SECOND = 2 * .000694 + 16 * .0000131 + 80 * .00000222  # 2 x A100-80GB + 16 CPU + 80 GiB list rate = 0.0017752
TIMEOUT_S = 900
CODE = ('scripts/workcell_recgen_objects.py', 'modal_apps/workcell_recgen_objects.py', 'scripts/workcell_recgen_worker.py',
        'fast_report/x7.py', 'fast_report/recgen_fast.py', 'scripts/workcell_extra_models.py', 'scripts/workcell_photo_objects.py')


@app.function(image=image, gpu='A100-80GB:2', cpu=16, memory=80 * 1024, volumes={'/cache': volumes['/cache']},
              timeout=TIMEOUT_S, retries=0, min_containers=0)
def generate(inputs: dict, records: list) -> dict:
    from scripts import workcell_recgen_objects as objects
    started = time.monotonic()
    timing, error, procs = {}, None, []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        try:
            for name, payload in inputs.items():
                if Path(name).name != name or not name.endswith('-input.npz'):
                    raise ValueError(f'unsafe input name {name!r}')
                (root / name).write_bytes(payload)
            for gpu, plan in enumerate(objects.plans(records)):
                path = root / f'recgen-plan-{gpu}.json'
                path.write_text(json.dumps(plan, indent=1))
                procs.append((gpu, _start(['/opt/recgen-venv/bin/python', '/repo/scripts/workcell_recgen_worker.py', str(path)], gpu)))
            workers = {}
            for gpu, proc in procs:
                workers[gpu] = json.loads(_finish(proc, f'GPU {gpu} RecGen', timeout=TIMEOUT_S - 150).strip().splitlines()[-1])
                (root / f'recgen-report-{gpu}.json').write_text(json.dumps(workers[gpu], indent=1))
            timing['recgenSeconds'] = time.monotonic() - started
            for record in records:
                if record['views']:
                    worker = next(w for w in workers.values() if record['kind'] in w['jobs'])
                    objects.place(root, record, worker['jobs'][record['kind']]['models'][objects.GROUP], worker['modelLoadSeconds'])
            timing['placementSeconds'] = time.monotonic() - started - timing['recgenSeconds']
        except Exception as exc:  # noqa: BLE001  returned with whatever finished; the caller fails loudly
            error = {'type': type(exc).__name__, 'message': str(exc)[-3000:]}
        finally:
            for _, proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode='w:gz') as archive:
            for path in sorted([*root.glob('*.glb'), *root.glob('*.json')]):
                archive.add(path, arcname=path.name)
    return {'archive': payload.getvalue(), 'containerWallSeconds': time.monotonic() - started, **timing, 'error': error}


def _safe(text):
    return '\n'.join(line for line in str(text).splitlines() if not re.search(r'capabilit|token|secret', line, re.I))


def ledger_append(destination, entry):
    """Append one call to OUT/spend-ledger.json (failed calls too) and refresh the totals; list-rate estimates only."""
    path = Path(destination) / 'spend-ledger.json'
    ledger = json.loads(path.read_text()) if path.is_file() else {
        'schemaVersion': 1, 'mode': 'ephemeral modal run', 'hardware': '2 x A100-80GB; 16 CPU; 80 GiB; one container per call',
        'usdPerSecond': USD_PER_SECOND, 'rateSource': 'https://modal.com/pricing', 'rateCheckedDate': '2026-10-01',
        'basis': ('reserved-resource list rate x seconds. functionSeconds: container wall inside the function; callSeconds: local '
                  'spawn to result (scheduling, cold start, image pull and transfer included); image build excluded; not an invoice'),
        'actualBilledUsd': None, 'calls': []}
    for key in ('functionSeconds', 'callSeconds'):
        entry[key.replace('Seconds', 'WindowEstimateUsd')] = None if entry.get(key) is None else round(entry[key] * USD_PER_SECOND, 4)
    ledger['calls'].append(entry)
    calls = ledger['calls']
    ledger['totals'] = {'calls': len(calls), 'failedCalls': sum(c['status'] != 'completed' for c in calls),
                        **{k: round(sum(c.get(k) or 0 for c in calls), 4) for k in
                           ('functionSeconds', 'callSeconds', 'functionWindowEstimateUsd', 'callWindowEstimateUsd')}}
    path.write_text(json.dumps(ledger, indent=2) + '\n')
    return ledger


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@app.local_entrypoint()
def main(run: str, scenes: str, out: str, sources: str = '', control_canonical: bool = False, label: str = ''):
    from scripts import workcell_recgen_objects as objects
    run_dir, destination = Path(run), Path(out)
    paths = [Path(p) for p in sources.split(',')] if sources else None
    if paths is not None and not all(p.is_file() for p in paths):
        raise ValueError('Every source photo must exist')
    destination.mkdir(parents=True, exist_ok=True)
    jobs, scene_specs = [], []
    for spec in scenes.split(';'):
        name, rest = spec.split('=')
        photos, ids = rest.split(':')
        photos, ids = [int(p) for p in photos.split(',')], ids.split(',')
        scene_specs.append({'scene': name, 'photos': photos, 'objectIds': ids})
        jobs += objects.build_inputs(run_dir, photos, ids, sources=paths, scene=name)
        if control_canonical:
            for job in objects.build_inputs(run_dir, photos, ids, sources=None, scene=name + '-canonical'):
                job['record']['control'] = 'canonical frames (no original-photo resampling); same objects, views and placement'
                jobs.append(job)
    records = [job['record'] for job in jobs]
    inputs = {job['record']['input']: objects.npz_bytes(job['arrays']) for job in jobs if job['record']['views']}
    entry = {'label': label or scenes + (' +canonical control' if control_canonical else ''), 'appId': app.app_id,
             'functionCallId': None, 'status': 'started', 'startedUnix': round(time.time(), 1),
             'inputMegabytes': round(sum(map(len, inputs.values())) / 1e6, 2), 'jobs': [r['kind'] for r in records if r['views']]}
    start, result = time.monotonic(), None
    try:
        call = generate.spawn(inputs, records)
        entry['functionCallId'] = call.object_id
        result = call.get()
        with tarfile.open(fileobj=io.BytesIO(result['archive']), mode='r:gz') as archive:
            for member in archive.getmembers():
                if Path(member.name).name != member.name or not member.isfile():
                    raise ValueError('Unexpected archive member')
            archive.extractall(destination, filter='data')
        entry.update({k: result.get(k) for k in ('recgenSeconds', 'placementSeconds')})
        if result['error']:
            raise RuntimeError(f"Cloud stage failed; finished artifacts kept: {_safe(result['error'])}")
        entry['status'] = 'completed'
    except Exception as exc:
        entry.update(status='failed', error=_safe(f'{type(exc).__name__}: {exc}')[-1500:])
        raise
    finally:
        entry['callSeconds'] = round(time.monotonic() - start, 2)
        entry['functionSeconds'] = None if result is None else round(result['containerWallSeconds'], 2)
        ledger_append(destination, entry)
    summary = []
    for job in jobs:
        record, kind = job['record'], job['record']['kind']
        if (destination / f'{kind}.json').is_file():
            record = json.loads((destination / f'{kind}.json').read_text())
            if paths is not None:
                objects.overlay(job['arrays'], record, destination / record['glb'], paths, destination / f'{kind}-overlay.jpg')
        summary.append({k: record.get(k) for k in objects.SUMMARY_KEYS})
    outcome = {'schemaVersion': 1, 'run': str(run_dir), 'scenes': scene_specs, 'acceptance': objects.ACCEPT, 'call': entry,
               'inputs': {'sourcesSha256': None if paths is None else {p.name: _sha(p) for p in paths},
                          'runFilesSha256': {n: _sha(run_dir / n) for n in ('objects.json', 'geometry.json', 'sam3.json')}},
               'implementationSha256': {c: _sha(Path(objects.__file__).resolve().parents[1] / c) for c in CODE},
               'objects': summary, 'basis': objects.BASIS}
    (destination / 'outcome.json').write_text(json.dumps(outcome, indent=2) + '\n')
    print(json.dumps({'status': entry['status'], 'appId': entry['appId'], 'callSeconds': entry['callSeconds'],
                      'objects': [(s['kind'], s['accepted']) for s in summary]}))
