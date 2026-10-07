"""Run the rebuild stages of scripts/workcell_rebuild_report.py (and its research harnesses) in the cloud, CPU only, ephemeral,
on a work tree in the panoptes-completion-ab volume (mounted at /data), so nothing numeric runs on the local machine.

The code is mounted at the same absolute paths as on the workstation (worktree, research notes, serving scripts and package,
platform package and scripts, the Pages build helpers), so every path default in those scripts holds; data paths point at
/data/... Upload inputs with `modal volume put panoptes-completion-ab LOCAL REMOTE`, read results with `modal volume get`.

  modal run modal_apps/workcell_rebuild_remote.py --cmd 'python WT/scripts/workcell_rebuild_report.py filter ...' [--env platform]
  modal run modal_apps/workcell_rebuild_remote.py --generate-spec SPEC.json   ({"run", "out", "config", "objects", "moge", "seed"})
  modal run modal_apps/workcell_rebuild_remote.py --assemble-spec SPEC.json   ({"run", "out", "objects", "variants"})
--env serving (default): PYTHONPATH = the serving package (its ehs_spatial); platform: cwd and PYTHONPATH = the platform
checkout (its ehs_spatial and scripts packages), for workcell_rebuild_publish.py trial.
--check-spec: a modal_apps/workcell_layer_trial.py task (box_faces, obvious_errors, ...) on a view, served files and layer in the volume.
--assemble-spec: completion_ab.py --stage assemble (the unchanged assembly) on the volume's run and candidates.
--generate: SAM 3D Objects (modal_apps/sam3d_research.py, unchanged, A100) on the inputs of workcell_rebuild_report.sam3d_input
(the photo, the object's own mask without other objects' pixels, the visibility-filtered MVS depth with MoGe-3 filling the holes
inside the mask only), every photo with a mask, written as completion_ab.py --stage sam3d writes them (sam3d/, sam3d-frame_*/).
"""
import json
import os
from pathlib import Path
import subprocess
import time

import modal

WT = Path('/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed')
NOTES = Path('/Users/adam/Desktop/panoptes-public/research-notes')
SERVING = Path('/Users/adam/Desktop/panoptes-public/panoptes-serving')
PAGES = Path('/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages')
PLATFORM = Path('/Users/adam/Desktop/Tesla/panoptes-platform')
CPU8 = 8 * .0000131 + 32 * .00000222  # 8 CPU, 32 GiB list rate (USD/s); not an invoice
LEDGER = Path('/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/fix090/ledger.jsonl')

app = modal.App('workcell-rebuild-remote')
vol = modal.Volume.from_name('panoptes-completion-ab')
skip = ['__pycache__', 'node_modules', '*.jpg', '*.png', 'dist*']
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('libgl1', 'libgomp1', 'libglib2.0-0', 'libx11-6', 'fonts-noto-cjk')
         .pip_install('numpy==2.5.1', 'scipy==1.18.0', 'opencv-python-headless==5.0.0.93', 'open3d==0.19.0', 'trimesh==5.1.0',
                      'pillow==12.3.0', 'matplotlib==3.11.0', 'shapely==2.1.2', 'pydantic==2.13.4', 'networkx==3.6.1'))
for local, remote in [(WT / 'scripts', WT / 'scripts'), (WT / 'configs', WT / 'configs'), (WT / 'fast_report', WT / 'fast_report'),
                      (WT / 'fast_report', '/opt/fr/fast_report'), (WT / 'modal_apps', WT / 'modal_apps'),
                      (NOTES / 'completion-ab-090-2026-10-06', NOTES / 'completion-ab-090-2026-10-06'),
                      (NOTES / 'completion-licence-ab-2026-10-05', NOTES / 'completion-licence-ab-2026-10-05'),
                      (SERVING / 'ehs_spatial', SERVING / 'ehs_spatial'), (SERVING / 'scripts/research', SERVING / 'scripts/research'),
                      (SERVING / 'scripts/research', '/serving/scripts/research'),
                      (PLATFORM / 'ehs_spatial', PLATFORM / 'ehs_spatial'), (PLATFORM / 'scripts', PLATFORM / 'scripts')]:
    image = image.add_local_dir(local, str(remote), ignore=skip)
for name in ('pack-model.py', 'build-unified-data.py'):
    image = image.add_local_file(PAGES / name, str(PAGES / name))

if True:  # the SAM 3D class (unchanged) joins this app so the CPU caller below can reach it from the cloud
    import sys
    sys.path.insert(0, str(WT / 'modal_apps'))
    import sam3d_research  # noqa: E402
    app.include(sam3d_research.app)


def environment(kind):
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8')
    env['PYTHONPATH'] = str(PLATFORM) if kind == 'platform' else str(SERVING)
    return env, (str(PLATFORM) if kind == 'platform' else '/data')


@app.function(image=image, volumes={'/data': vol}, cpu=8, memory=32 * 1024, timeout=3600, retries=0, min_containers=0)
def sh(cmd: str, kind: str = 'serving') -> dict:
    started = time.monotonic()
    vol.reload()
    env, cwd = environment(kind)
    done = subprocess.run(['bash', '-c', cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=3500)
    vol.commit()
    return dict(returncode=done.returncode, stdout=done.stdout[-12000:], stderr=done.stderr[-6000:], seconds=time.monotonic() - started)


@app.function(image=image, volumes={'/data': vol}, cpu=4, memory=16 * 1024, timeout=3600, retries=0, min_containers=0)
def generate(spec: dict) -> dict:
    """spec: run, out, objects [ids], moge (dir of moge-<frame>.npz), seed (42), frames ('all')."""
    import sys
    import numpy as np
    started = time.monotonic()
    vol.reload()
    sys.path[:0] = [str(WT / 'scripts'), str(SERVING / 'scripts/research')]
    import workcell_rebuild_report as wr
    run, out = Path(spec['run']), Path(spec['out'])
    objects = {o['object_id']: o for o in json.loads((run / 'evidence/objects.json').read_text())['objects']}
    cfg = json.loads(Path(spec['config']).read_text())
    model = sam3d_research.SAM3DObjects()
    done = []
    for oid in spec['objects']:
        obj = objects[oid]
        default = max(obj['views'], key=lambda v: v['mask_pixels'])['frame_id']
        for view in sorted(obj['views'], key=lambda v: v['frame_id'] != default):
            fid = view['frame_id']; dest = out / ('sam3d' if fid == default else 'sam3d-' + fid)
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / f'{oid}.npz').exists():
                continue
            rgb, mask, pointmap, meta = wr.sam3d_input(run, cfg, objects, oid, fid, Path(spec['moge']))
            t = time.monotonic()
            res = model.run.remote(rgb, mask, pointmap, spec.get('seed', 42))
            meta['call_seconds'] = time.monotonic() - t
            if 'error' in res:
                meta.update(error=res['error'][-3000:], seconds=res['seconds'])
            else:
                np.savez_compressed(dest / f'{oid}.npz', vertices=res['vertices'], faces=res['faces'].astype(np.int32), colors=res['colors'],
                                    object_to_camera_p3d=res['objectToCamera'])
                meta.update(seconds=res['seconds'], gpu=res['gpu'], pins=res['pins'], vertices=len(res['vertices']), faces=len(res['faces']))
            log = json.loads((dest / 'record.json').read_text()) if (dest / 'record.json').exists() else {'objects': {}}
            log['objects'][oid] = meta
            (dest / 'record.json').write_text(json.dumps(log, indent=1))
            vol.commit()
            done.append(dict(object=oid, frame=fid, gpuSeconds=meta.get('seconds'), error='error' in meta, coverage=meta['mask_mvs_pixels'] / max(1, meta['mask_pixels']),
                             filled=meta['mask_depth_pixels'] / max(1, meta['mask_pixels'])))
    return dict(done=done, seconds=time.monotonic() - started)


@app.function(image=image, volumes={'/data': vol}, cpu=8, memory=32 * 1024, timeout=3600, retries=0, min_containers=0)
def assemble(spec: dict) -> dict:
    """completion_ab.py --stage assemble (unchanged assembly) in this container: spec run, out, objects, variants."""
    import sys
    import types
    started = time.monotonic()
    vol.reload()
    os.environ.update(AB_RUN=spec['run'], AB_OUT=spec['out'], AB_OBJECTS=','.join(spec['objects']))
    sys.path.insert(0, str(NOTES / 'completion-licence-ab-2026-10-05'))
    import completion_ab as ab
    body = ab.assemble_variants.local
    ab.assemble_variants = types.SimpleNamespace(remote=lambda *a, **k: body(*a, **k))  # the same function, run here
    ab.stage_assemble(spec['variants'])
    vol.commit()
    return dict(seconds=time.monotonic() - started)


@app.function(image=image, volumes={'/data': vol}, cpu=8, memory=32 * 1024, timeout=3600, retries=0, min_containers=0)
def check(spec: dict) -> dict:
    """modal_apps/workcell_layer_trial.py's run, with the view, photos, served files and layer read from the volume: spec view
    (a publication view JSON), run (its input photos by camera), served (dir: api/api/assets/<id>, api/blobs/<sha>, or ''),
    api (the origin the assets come from: https://layer-trial.invalid/report/api for a trial, the publication service live),
    layer ('' or a layer JSON whose asset files sit next to it), task (a module path with run(ctx, opts)), opts, out (dir)."""
    import io
    import sys
    import types
    import urllib.request
    started = time.monotonic()
    vol.reload()
    sys.path[:0] = [str(WT / 'scripts')]
    import workcell_shape_check as wsc
    base = 'https://layer-trial.invalid/report/'
    served = Path(spec['served']) if spec.get('served') else None
    layer = Path(spec['layer']) if spec.get('layer') else None
    real = urllib.request.urlopen

    def urlopen(url, *a, **k):
        u = url if isinstance(url, str) else url.full_url
        if u.startswith(base):
            rel = u[len(base):]
            if layer is not None and rel.startswith('measurement-layer/'):
                return io.BytesIO((layer if rel.endswith('.json') else layer.parent / Path(rel).name).read_bytes())
            return io.BytesIO((served / rel).read_bytes())
        return real(url, *a, **k)
    urllib.request.urlopen = urlopen  # ponytail: load_report fetches with urllib.request.urlopen; nothing else is patched
    view = json.loads(Path(spec['view']).read_text()); doc = view['publication']['snapshot']['revision']['document']
    run = Path(spec['run']); frames = {f['frame_id']: f for f in json.loads((run / 'manifest.json').read_text())['frames']}
    photos = {c['imageId']: (run / frames[Path(c['sourceRefs'][0]['sourceCameraId']).name]['input']).read_bytes() for c in doc['cameras']}
    layer_url = base + 'measurement-layer/' + (json.loads(layer.read_text())['publicationId'] + '.json') if layer else ''
    ctx = wsc.load_report(json.dumps(view).encode(), photos, layer_url, spec['api'])
    module = types.ModuleType('trial_task'); module.__file__ = spec['task']
    exec(compile(Path(spec['task']).read_text(), spec['task'], 'exec'), module.__dict__)
    result = module.run(ctx, spec.get('opts') or {})
    out = Path(spec['out']); out.mkdir(parents=True, exist_ok=True)
    for name, data in (result.pop('files', None) or {}).items():
        (out / name).write_bytes(data)
    record = dict(result=result, layerRevision=(ctx['layer'] or {}).get('revisionId'), layerModels=sorted(((ctx['layer'] or {}).get('models') or {})),
                  nativeToMeters=ctx['S'], skipped=ctx['skipped'], objects={o['id']: o['label'] for o in ctx['objects']}, containerSeconds=time.monotonic() - started)
    (out / 'results.json').write_text(json.dumps(record, indent=1, ensure_ascii=False, default=float) + '\n')
    vol.commit()
    return dict(seconds=time.monotonic() - started, out=str(out))


def journal(entry):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a') as stream:
        stream.write(json.dumps(entry) + '\n')


@app.local_entrypoint()
def main(cmd: str = '', env: str = 'serving', generate_spec: str = '', assemble_spec: str = '', check_spec: str = '', label: str = ''):
    start = time.monotonic()
    if check_spec:
        result = check.remote(json.loads(Path(check_spec).read_text()))
        journal(dict(label=label or 'check', kind='cpu8', seconds=result['seconds'], estimateUsd=result['seconds'] * CPU8))
        print(json.dumps(result))
        return
    if assemble_spec:
        result = assemble.remote(json.loads(Path(assemble_spec).read_text()))
        journal(dict(label=label or 'assemble', kind='cpu8', seconds=result['seconds'], estimateUsd=result['seconds'] * CPU8))
        print(json.dumps(result))
        return
    if generate_spec:
        spec = json.loads(Path(generate_spec).read_text())
        result = generate.remote(spec)
        gpu = sum(d['gpuSeconds'] or 0 for d in result['done'])
        cost = gpu * sam3d_research.USD_PER_SECOND + result['seconds'] * (4 * .0000131 + 16 * .00000222)
        journal(dict(label=label or 'generate', kind='sam3d', calls=len(result['done']), gpuSeconds=gpu, cpuSeconds=result['seconds'], estimateUsd=cost))
        print(json.dumps(result, indent=1))
        return
    result = sh.remote(cmd, env)
    journal(dict(label=label or cmd[:80], kind='cpu8', seconds=result['seconds'], estimateUsd=result['seconds'] * CPU8, returncode=result['returncode']))
    print(result['stdout'])
    if result['stderr'].strip():
        print('--- stderr ---\n' + result['stderr'])
    print(json.dumps({'returncode': result['returncode'], 'seconds': round(result['seconds'], 1), 'callSeconds': round(time.monotonic() - start, 1)}))
