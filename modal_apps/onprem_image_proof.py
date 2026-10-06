"""Proof of the on-prem images: build docker/*.Dockerfile on Modal (modal.Image.from_dockerfile, ephemeral) and run the pipeline's
stages inside them through scripts/onprem/run_stage.py with every MODAL_* variable removed and the container network blocked,
as `docker run --network none` would on a customer server. Modal is only the test bench here; nothing is deployed.

  ONPREM_PROOF=cpu    modal run modal_apps/onprem_image_proof.py --view V --photos-dir D --photo ID=F,.. --layer-url U --api A \
                          --bundle BUNDLE --reference FLOOR_RESULTS.json --out NEW_DIR      self-tests + offline floor check
  ONPREM_PROOF=fetch  modal run modal_apps/onprem_image_proof.py --out NEW_DIR           fetch_weights.py -> volume (network on)
  ONPREM_PROOF=gpu    modal run modal_apps/onprem_image_proof.py --view .. --bundle .. --run-dir RUN [--moge-reference J] --out NEW_DIR
                          one A100, network blocked: Pi3X on the run's photos, the MoGe-3 check, SAM 3 'floor' text prompts
  ONPREM_PROOF=recgen modal run modal_apps/onprem_image_proof.py --out NEW_DIR           build + import check of docker/recgen.Dockerfile

BUNDLE = scripts/onprem/run_stage.py --record output (every URL the floor check read). Inputs travel as call arguments.
"""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

import modal

REPO = Path(__file__).resolve().parents[1]
SERVING = Path(os.environ.get('PANOPTES_SERVING', '/Users/adam/Desktop/panoptes-public/panoptes-serving'))
STAGE = os.environ.get('ONPREM_PROOF', 'cpu')
CPU_RATE = 8 * .0000131 + 16 * .00000222  # list rates (USD/s), not an invoice
GPU_RATE = .000694 + 8 * .0000131 + 32 * .00000222
FETCH_RATE = 4 * .0000131 + 8 * .00000222
WEIGHTS = 'panoptes-onprem-weights'
DOCKERFILE = {'cpu': 'workcell-cpu', 'fetch': 'workcell-gpu', 'gpu': 'workcell-gpu', 'recgen': 'recgen'}[STAGE]


def build_context() -> Path:
    """SRC/workcell = this repository, SRC/serving = panoptes-serving: code only (what the Dockerfiles COPY)."""
    ctx = Path(tempfile.mkdtemp(prefix='panoptes-onprem-context-'))  # one per run: parallel runs must not share it
    skip = shutil.ignore_patterns('__pycache__', '*.pyc', 'node_modules', 'outputs')
    for src, dst in ((REPO / 'docker', 'workcell/docker'), (REPO / 'scripts', 'workcell/scripts'), (REPO / 'modal_apps', 'workcell/modal_apps'),
                     (SERVING / 'scripts', 'serving/scripts'), (SERVING / 'modal_apps', 'serving/modal_apps'), (SERVING / 'ehs_spatial', 'serving/ehs_spatial')):
        shutil.copytree(src, ctx / dst, ignore=skip)
    return ctx


if modal.is_local():
    import atexit
    CTX = build_context()
    atexit.register(shutil.rmtree, CTX, True)  # the staged copy is only needed while this run uploads it
    image = (modal.Image.from_dockerfile(CTX / 'workcell/docker' / f'{DOCKERFILE}.Dockerfile', context_dir=CTX)
             .env({'ONPREM_PROOF': STAGE}))  # the container imports this module too and must define the same stage
else:
    image = modal.Image.debian_slim()
app = modal.App(f'onprem-image-proof-{STAGE}')


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL')}  # no Modal identity reaches the stage
    env.update({'HOME': '/tmp/home', 'HF_HUB_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1', **extra})
    Path('/tmp/home').mkdir(exist_ok=True)
    return env


def sh(name, cmd, cwd='/workcell', timeout=1500, env=None):
    start = time.monotonic()
    p = subprocess.run(cmd, cwd=cwd, env=env or clean_env(), capture_output=True, text=True, timeout=timeout)
    return dict(step=name, cmd=' '.join(map(str, cmd)), rc=p.returncode, seconds=round(time.monotonic() - start, 2),
                stdout=p.stdout[-6000:], stderr=p.stderr[-6000:])


PROBE = ("import socket\n"
         "out = {}\n"
         "for name, f in (('tcp 1.1.1.1:443', lambda: socket.create_connection(('1.1.1.1', 443), timeout=5)),\n"
         "                ('dns huggingface.co', lambda: socket.getaddrinfo('huggingface.co', 443))):\n"
         "    try:\n        f(); out[name] = 'REACHABLE'\n"
         "    except Exception as e:\n        out[name] = 'blocked: ' + type(e).__name__\n"
         "print(out)")


def unpack(blob: bytes, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as t:
        t.extractall(dest, filter='data')


def pack(path: Path, arcname: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as t:
        t.add(path, arcname=arcname)
    return buf.getvalue()


def write_inputs(inputs: dict) -> Path:
    root = Path('/tmp/in'); (root / 'photos').mkdir(parents=True, exist_ok=True)
    (root / 'view.json').write_bytes(inputs['view'])
    for name, data in inputs['photos'].items():
        (root / 'photos' / name).write_bytes(data)
    unpack(inputs['bundle'], root)
    return root


def check_args(root: Path, inputs: dict, out: Path):
    return ['--view', root / 'view.json', '--photos-dir', root / 'photos', '--photo', inputs['photo'],
            '--layer-url', inputs['layer_url'], '--api', inputs['api'], '--out', out]


if STAGE == 'cpu':
    @app.function(image=image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0, block_network=True)
    def cpu_proof(inputs: dict) -> dict:
        start = time.monotonic()
        root = write_inputs(inputs)
        py, asm = '/opt/checks/bin/python', '/opt/assemble/bin/python'
        steps = [sh('network probe', [py, '-c', PROBE]),
                 sh('run_stage self-test', [py, 'scripts/onprem/run_stage.py', '--self-test']),
                 sh('fetch_weights self-test', [py, 'scripts/onprem/fetch_weights.py', '--self-test']),
                 sh('shape core self-test', [py, '/check/shape_core.py']),
                 sh('check modules self-tests', [py, '-c', "import sys; sys.path.insert(0, '/check'); import importlib\n"
                     "for n in ('floor', 'lines', 'plane_stereo', 'transfer'):\n    importlib.import_module('workcell_checks.' + n)._check(); print(n, 'ok')"]),
                 sh('moge check self-test (CPU part)', [py, 'modal_apps/workcell_moge_check.py']),
                 sh('assemble venv imports', [asm, '-c', "import sys; sys.path.insert(0, '/serving/scripts/research'); import assemble_lucida_scene, "
                     "numpy, open3d, trimesh, cv2, scipy, PIL; print('assemble_lucida_scene', numpy.__version__, open3d.__version__, trimesh.__version__)"]),
                 sh('floor check offline', [py, 'scripts/onprem/run_stage.py', '--offline', root / 'bundle', 'modal_apps/workcell_view_checks.py',
                                            '--checks', 'floor', *check_args(root, inputs, Path('/tmp/out/floor'))])]
        freeze = dict(checks=subprocess.run([py, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True).stdout,
                      assemble=subprocess.run(['uv', 'pip', 'freeze', '--python', asm], capture_output=True, text=True).stdout)
        result = Path('/tmp/out/floor/results.json')
        return dict(steps=steps, floor=json.loads(result.read_text()) if result.exists() else None, freeze=freeze,
                    containerSeconds=time.monotonic() - start)


if STAGE == 'fetch':
    @app.function(image=image, cpu=4, memory=8 * 1024, timeout=1800, retries=0, min_containers=0,
                  volumes={'/weights': modal.Volume.from_name(WEIGHTS, create_if_missing=True),
                           '/v/src-sam3': modal.Volume.from_name('sam3-hf-cache')})
    def fetch(models: str, sam3_source: str) -> dict:
        start = time.monotonic()
        py = '/opt/pi3x/bin/python'
        hub = [m for m in models.split(',') if m != 'sam3']
        env = clean_env(HF_HUB_OFFLINE='0')
        steps = [sh('fetch from huggingface.co', [py, 'scripts/onprem/fetch_weights.py', '--cache', '/weights', '--models', ','.join(hub)], env=env)] if hub else []
        if 'sam3' in models.split(','):
            steps.append(sh('sam3 from an existing HF cache', [py, 'scripts/onprem/fetch_weights.py', '--cache', '/weights', '--models', 'sam3',
                                                              '--source', sam3_source], env=env))
        steps.append(sh('verify (offline re-hash)', [py, 'scripts/onprem/fetch_weights.py', '--cache', '/weights', '--verify']))
        modal.Volume.from_name(WEIGHTS).commit()
        manifest = Path('/weights/manifest.json')
        return dict(steps=steps, manifest=json.loads(manifest.read_text()) if manifest.exists() else None, containerSeconds=time.monotonic() - start)


if STAGE == 'gpu':
    @app.function(image=image, gpu='A100-80GB', cpu=8, memory=32 * 1024, timeout=1800, retries=0, min_containers=0, block_network=True,
                  volumes={'/weights': modal.Volume.from_name(WEIGHTS)})
    def gpu_proof(inputs: dict) -> dict:
        start = time.monotonic()
        root = write_inputs(inputs)
        unpack(inputs['run'], Path('/tmp'))  # /tmp/run030: manifest + canonical frames (fresh Pi3X) and the original geometry
        fresh = Path('/tmp/pi3x-run'); (fresh / 'evidence').mkdir(parents=True)
        shutil.copy(Path('/tmp/run030/manifest.json'), fresh / 'manifest.json')
        shutil.copytree(Path('/tmp/run030/evidence/canonical'), fresh / 'evidence/canonical')
        run_stage = 'scripts/onprem/run_stage.py'
        steps = [sh('network probe', ['/opt/pi3x/bin/python', '-c', PROBE]),
                 sh('nvidia-smi', ['nvidia-smi', '-L']),
                 sh('pi3x offline', ['/opt/pi3x/bin/python', run_stage, '--weights', '/weights', '/serving/modal_apps/pi3x_geometry.py', '--run', fresh])]
        steps.append(sh('moge-3 check offline', ['/opt/moge/bin/python', run_stage, '--weights', '/weights', '--offline', root / 'bundle',
                                                 'modal_apps/workcell_moge_check.py', *check_args(root, inputs, Path('/tmp/out/moge'))[:-2],
                                                 '--run-dir', '/tmp/run030', '--out', '/tmp/out/moge']))
        sam = Path('/tmp/sam'); sam.mkdir()
        (sam / 'words.json').write_text(json.dumps(['floor']))
        frames = json.loads(Path('/tmp/run030/manifest.json').read_text())['frames']
        for i, fr in enumerate(frames, 1):  # the worker's second pass (cart boxes) gets none: text prompts only
            shutil.copy(root / 'photos' / Path(fr['input']).name, sam / f'source-{i}.jpg')
            shutil.copy(Path('/tmp/run030') / fr['canonical'], sam / f'photo-{i}.png')
        (sam / 'cart-boxes.json').write_text(json.dumps({'results': [[] for _ in frames]}))
        steps.append(sh('sam3 floor text prompt offline', ['/opt/sam3/bin/python', run_stage, '--weights', '/weights', 'scripts/workcell_sam_worker.py', sam]))
        # Outputs go back through the volume: a return value above Modal's inline size is uploaded over the (blocked) network.
        box = Path('/tmp/box'); box.mkdir()
        if (fresh / 'geometry').exists():
            shutil.copytree(fresh / 'geometry', box / 'geometry')
        for src, name in ((Path('/tmp/out/moge/results.json'), 'moge.json'), (sam / 'sam3.json', 'sam3.json'), (sam / 'sam-timing.json', 'samTiming.json')):
            if src.exists():
                shutil.copy(src, box / name)
        key = f'proof/{os.urandom(8).hex()}.tar.gz'
        (Path('/weights') / key).parent.mkdir(exist_ok=True)
        (Path('/weights') / key).write_bytes(pack(box, 'box'))
        modal.Volume.from_name(WEIGHTS).commit()
        return dict(steps=steps, volumeKey=key, containerSeconds=time.monotonic() - start)


if STAGE == 'recgen':
    @app.function(image=image, cpu=2, memory=8 * 1024, timeout=1800, retries=0, min_containers=0, block_network=True)
    def recgen_proof() -> dict:
        start = time.monotonic()
        steps = [sh('network probe', ['python', '-c', PROBE], cwd='/serving'),
                 sh('recgen imports', ['python', '-c', "import xformers.ops as xops; assert xops.fmha.BlockDiagonalMask; from recgen_inference import "
                     "build_recgen, generate, generate_multiview; import torch, spconv; print('RecGen import ready', torch.__version__)"], cwd='/serving'),
                 sh('run_stage self-test', ['python', '/workcell/scripts/onprem/run_stage.py', '--self-test'], cwd='/serving'),
                 sh('lucida app loads in-process', ['python', '-c', "import sys; sys.path[:0] = ['/workcell/scripts/onprem', '/serving']; import run_stage; "
                     "m = run_stage.load_app(__import__('pathlib').Path('/serving/modal_apps/lucida_assets.py')); "
                     "print(sorted(m.app.registered_functions), m.CODE_REV, m.MODEL_REV)"], cwd='/serving')]
        freeze = subprocess.run(['uv', 'pip', 'freeze', '--python', '/opt/recgen-py/bin/python'], capture_output=True, text=True).stdout
        return dict(steps=steps, freeze=freeze, containerSeconds=time.monotonic() - start)


# ---------------------------------------------------------------- local side: inputs, comparisons, ledger
def bundle_tar(bundle: str) -> bytes:
    return pack(Path(bundle), 'bundle')


def ledger(out: Path, seconds: float, rate: float, hardware: str, call: float):
    row = dict(mode='ephemeral modal run (proof bench only)', hardware=hardware, functionSeconds=seconds, callSeconds=call,
               estimateUsd=rate * seconds, actualBilledUsd=None, rateSource='https://modal.com/pricing',
               note='callSeconds includes image build/pull and cold start; billed time can be higher than functionSeconds')
    (out / 'spend-ledger.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


def floor_diff(ref: dict, new: dict) -> dict:
    a, b = ref['results']['floor'], new['results']['floor']
    keys = sorted(set(a['objects']) | set(b['objects']))
    return dict(sameObjects=set(a['objects']) == set(b['objects']),
                maxAbsDiffM=max(abs(a['objects'][k][q] - b['objects'][k][q]) for k in keys for q in ('bottomM', 'topM')),
                statusChanges={k: (a['objects'][k]['status'], b['objects'][k]['status']) for k in keys if a['objects'][k]['status'] != b['objects'][k]['status']},
                tolM=(a.get('tolM'), b.get('tolM')), sameScale=ref['nativeToMeters'] == new['nativeToMeters'],
                sameLayerRevision=ref['layerRevision'] == new['layerRevision'])


def rle_decode(text: str):
    import numpy as np
    r = json.loads(text); h, w = r['size']; flat = np.zeros(h * w, bool); pos, val = 0, False
    for c in r['counts']:
        flat[pos:pos + c] = val; pos += c; val = not val
    return flat.reshape((h, w), order='F')


def compare_gpu(result: dict, run_dir: Path, out: Path, moge_reference: str) -> dict:
    import numpy as np
    from PIL import Image
    import cv2
    cmp = {}
    if 'geometry' in result:
        unpack(result.pop('geometry'), out / 'pi3x')
        rows = {}
        for g in sorted((out / 'pi3x/geometry/frames').iterdir()):
            o = run_dir / 'geometry/frames' / g.name
            pose_ref = o / 'camera_to_world.pi3x-bf16.npy' if (o / 'camera_to_world.pi3x-bf16.npy').exists() else o / 'camera_to_world.npy'
            P, Pr = np.load(g / 'camera_to_world.npy'), np.load(pose_ref)
            pts, ptsr = np.load(g / 'pts3d.npy'), np.load(o / 'pts3d.npy')
            valid = np.load(o / 'valid_mask.npy') & np.load(g / 'valid_mask.npy')
            d = np.linalg.norm(pts - ptsr, axis=-1)[valid]
            R = P[:3, :3] @ Pr[:3, :3].T
            rows[g.name] = dict(poseReference=pose_ref.name, poseMaxAbsEntry=float(np.abs(P - Pr).max()),
                                rotationDeg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                                centreDistNative=float(np.linalg.norm(P[:3, 3] - Pr[:3, 3])),
                                intrinsicsMaxAbs=float(np.abs(np.load(g / 'intrinsics.npy') - np.load(o / 'intrinsics.npy')).max()),
                                pointsMaxNative=float(d.max()), pointsMedianNative=float(np.median(d)), pointsP99Native=float(np.percentile(d, 99)),
                                validMaskIdentical=bool((np.load(o / 'valid_mask.npy') == np.load(g / 'valid_mask.npy')).all()),
                                confMaxAbs=float(np.abs(np.load(g / 'conf.npy') - np.load(o / 'conf.npy')).max()),
                                bitIdenticalPoints=bool(np.array_equal(pts, ptsr)))
        new_meta = json.loads((out / 'pi3x/geometry/candidate_manifest.json').read_text())
        old_meta = json.loads((run_dir / 'geometry/candidate_manifest.json').read_text())
        import hashlib
        glb = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        cmp['pi3x'] = dict(frames=rows, metric=(old_meta.get('native_metric_factor'), new_meta.get('native_metric_factor')),
                           pointCloudGlbSha256=(glb(run_dir / 'geometry/point_cloud.glb'), glb(out / 'pi3x/geometry/point_cloud.glb')),
                           pins={k: (old_meta.get(k), new_meta.get(k)) for k in ('model_revision', 'weights_sha256', 'code_revision', 'torch_version', 'device')})
    if 'moge' in result and moge_reference:
        ref, new = json.loads(Path(moge_reference).read_text())['reports']['030'], result['moge']
        (out / 'moge-results.json').write_text(json.dumps(new, indent=1, ensure_ascii=False))
        key = lambda r: (r['entityId'], r['photo'])
        old_rows = {key(r): r for r in ref['objects']}
        both = [(old_rows[key(r)], r) for r in new['objects'] if key(r) in old_rows]
        res = lambda r, f: r[f]['median'] if r[f]['n'] else float('nan')
        cmp['moge'] = dict(pooledMedian=(ref['scale']['report']['pooled']['median'], new['scale']['report']['pooled']['median']),
                           photoMedians=(ref['scale']['report']['photoMedians'], new['scale']['report']['photoMedians']),
                           objectRows=(len(ref['objects']), len(new['objects']), len(both)),
                           statusChanges=[[*key(b), a['status'], b['status']] for a, b in both if a['status'] != b['status']],
                           maxAbsResidualDiff=max((abs(res(a, 'residual') - res(b, 'residual')) for a, b in both if a['residual']['n'] and b['residual']['n']), default=None))
    if 'sam3' in result:
        manifest = json.loads((run_dir / 'manifest.json').read_text())
        rows = {}
        for i, (view, fr) in enumerate(zip(result['sam3']['results'], manifest['frames']), 1):
            masks = [rle_decode(t) for t in view[0]['rle']]
            union = np.any(masks, axis=0) if masks else None
            A = np.asarray(fr['input_to_canonical_pixel_centres'], float)
            canon = cv2.warpAffine(union.astype(np.float32), A[:2], (518, 518), flags=cv2.INTER_LINEAR) > .5 if union is not None else np.zeros((518, 518), bool)
            ref_mask = np.asarray(Image.open(run_dir / 'evidence/floor-masks' / f"{fr['frame_id']}.png")) > 0
            x0, y0, x1, y1 = fr['content_rect_xyxy']; inside = np.zeros_like(ref_mask); inside[y0:y1, x0:x1] = True
            a, b = canon & inside, ref_mask & inside
            rows[fr['frame_id']] = dict(instances=len(masks), scores=view[0]['scores'], iouVsProductRunFloorMask=float((a & b).sum() / max(1, (a | b).sum())),
                                        newPixels=int(a.sum()), productRunPixels=int(b.sum()))
            Image.fromarray(canon.astype(np.uint8) * 255).save(out / f"sam3-floor-{fr['frame_id']}.png")
        cmp['sam3Floor'] = rows
    return cmp


@app.local_entrypoint()
def main(out: str, view: str = '', photos_dir: str = '', photo: str = '', layer_url: str = '', api: str = '', bundle: str = '',
         reference: str = '', run_dir: str = '', moge_reference: str = '', models: str = 'pi3x,moge3,sam3',
         sam3_source: str = '/v/src-sam3/huggingface/hub'):
    dest = Path(out)
    if dest.exists():
        raise ValueError('Choose a fresh output directory')
    dest.mkdir(parents=True)
    start = time.monotonic()
    if STAGE in ('cpu', 'gpu'):
        names = dict(item.split('=', 1) for item in photo.split(','))
        inputs = dict(view=Path(view).read_bytes(), photos={n: (Path(photos_dir) / n).read_bytes() for n in names.values()}, photo=photo,
                      layer_url=layer_url, api=api, bundle=bundle_tar(bundle))
    if STAGE == 'cpu':
        result = cpu_proof.remote(inputs)
        if result['floor'] and reference:
            result['comparison'] = floor_diff(json.loads(Path(reference).read_text()), result['floor'])
        result['ledger'] = ledger(dest, result['containerSeconds'], CPU_RATE, '8 CPU, 16 GiB, block_network', time.monotonic() - start)
    elif STAGE == 'fetch':
        result = fetch.remote(models, sam3_source)
        result['ledger'] = ledger(dest, result['containerSeconds'], FETCH_RATE, '4 CPU, 8 GiB, network on (fetch only)', time.monotonic() - start)
    elif STAGE == 'gpu':
        run = Path(run_dir).resolve()
        with tempfile.TemporaryDirectory() as tmp:  # the run's manifest, canonical frames and original Pi3X geometry (no outputs of later stages)
            stage = Path(tmp) / 'run030'
            (stage / 'evidence').mkdir(parents=True)
            shutil.copy(run / 'manifest.json', stage / 'manifest.json')
            shutil.copytree(run / 'evidence/canonical', stage / 'evidence/canonical')
            for f in (run / 'geometry/frames').iterdir():
                (stage / 'geometry/frames' / f.name).mkdir(parents=True)
                for name in ('camera_to_world.npy', 'pts3d.npy', 'valid_mask.npy', 'content_valid_mask.npy'):
                    shutil.copy(f / name, stage / 'geometry/frames' / f.name / name)
            inputs['run'] = pack(stage, 'run030')
        result = gpu_proof.remote(inputs)
        volume = modal.Volume.from_name(WEIGHTS)
        with tempfile.TemporaryDirectory() as tmp:
            unpack(b''.join(volume.read_file(result['volumeKey'])), Path(tmp))
            box = Path(tmp) / 'box'
            if (box / 'geometry').exists():
                result['geometry'] = pack(box / 'geometry', 'geometry')
            for name in ('moge', 'sam3', 'samTiming'):
                if (box / f'{name}.json').exists():
                    result[name] = json.loads((box / f'{name}.json').read_text())
        volume.remove_file(result['volumeKey'])
        result['comparison'] = compare_gpu(result, run, dest, moge_reference)
        result.pop('sam3', None)  # full-resolution RLE; the canonical masks are saved as sam3-floor-*.png
        result['ledger'] = ledger(dest, result['containerSeconds'], GPU_RATE, 'A100-80GB, 8 CPU, 32 GiB, block_network', time.monotonic() - start)
    else:
        result = recgen_proof.remote()
        result['ledger'] = ledger(dest, result['containerSeconds'], 2 * .0000131 + 8 * .00000222, '2 CPU, 8 GiB, block_network', time.monotonic() - start)
    (dest / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + '\n')
    for s in result.get('steps', []):
        print(f"[{'ok' if s['rc'] == 0 else 'FAIL rc=' + str(s['rc'])}] {s['step']} {s['seconds']}s :: {(s['stdout'].strip().splitlines() or [''])[-1][:300]}")
        if s['rc']:
            print('   stderr:', s['stderr'][-1500:])
    print(json.dumps({k: result[k] for k in ('comparison', 'ledger') if k in result}, default=str)[:4000])
