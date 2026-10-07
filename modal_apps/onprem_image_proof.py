"""Proof of the on-prem images: build docker/*.Dockerfile on Modal (modal.Image.from_dockerfile, ephemeral) and run the pipeline's
stages inside them through scripts/onprem/run_stage.py with every MODAL_* variable removed and the container network blocked,
as `docker run --network none` would on a customer server. Modal is only the test bench here; nothing is deployed.

  ONPREM_PROOF=cpu    modal run modal_apps/onprem_image_proof.py --view V --photos-dir D --photo ID=F,.. --layer-url U --api A \
                          --bundle BUNDLE --reference FLOOR_RESULTS.json [--run-dir RUN --targets ID=post,.. \
                          --clearance-reference CLEARANCE_RESULTS.json] --out NEW_DIR
                          self-tests, no modal client, stage apps under the stub, offline floor check [+ offline clearance B]
  ONPREM_PROOF=fetch  modal run modal_apps/onprem_image_proof.py --out NEW_DIR           fetch_weights.py -> volume (network on)
  ONPREM_PROOF=gpu    modal run modal_apps/onprem_image_proof.py --view .. --bundle .. --run-dir RUN [--moge-reference J] \
                          [--steps pi3x,moge,sam3] --out NEW_DIR
                          one A100, network blocked: Pi3X on the run's photos, the MoGe-3 check, SAM 3 'floor' text prompts
  ONPREM_PROOF=recgen modal run modal_apps/onprem_image_proof.py --out NEW_DIR           build + import check of docker/recgen.Dockerfile
  ONPREM_PROOF=recgen-generate modal run modal_apps/onprem_image_proof.py --run-dir RUN --object-id left_post [--replay] \
                          [--no-fetch] --out NEW_DIR
                          docker/recgen.Dockerfile: fetch_weights.py --models recgen into the volume (network on), then one A100
                          with the network blocked runs the RecGen check and generate_lucida_assets.py for that object through
                          run_stage.py --weights, with a gpu-budget.json that is already over the Modal limit (on-prem the gate is
                          off); the mesh is compared with RUN/generation/OBJECT (seed 42); the weights must hold no job copy after
  ONPREM_PROOF=sam3d  modal run modal_apps/onprem_image_proof.py --inputs NPZ_DIR [--reference REF_NPZ_DIR] --out NEW_DIR
                          docker/sam3d.Dockerfile on one A100 with the network ON: a Python socket / subprocess audit around
                          torch.hub (without and with run_stage.py) and around SAM 3D generation of every NPZ_DIR/*.npz
                          (rgb, mask, pointmap; seed 42) through run_stage.py --weights; weights from panoptes-onprem-sam3d
  ONPREM_PROOF=estop  modal run modal_apps/onprem_image_proof.py --photos-dir D --photo FILE,.. --reference MODAL_OUT --out NEW_DIR
                          docker/workcell-gpu.Dockerfile on one L4, network blocked: modal_apps/workcell_estop_mask.py through
                          run_stage.py --weights (/opt/sam3) with MODAL_OUT's text/tile/threshold/keep; every candidate mask is
                          compared with MODAL_OUT (the same app's `modal run` output)
PANOPTES_SERVING = the panoptes-serving checkout (the image's serving/ half of the build context); required.

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
SERVING = os.environ.get('PANOPTES_SERVING', '')  # no machine default: scripts/onprem/stage_context.sh SRC SERVING_DIR
STAGE = os.environ.get('ONPREM_PROOF', 'cpu')
CPU_RATE = 8 * .0000131 + 16 * .00000222  # list rates (USD/s), not an invoice
GPU_RATE = .000694 + 8 * .0000131 + 32 * .00000222
FETCH_RATE = 4 * .0000131 + 8 * .00000222
WEIGHTS = 'panoptes-onprem-weights'
DOCKERFILE = {'cpu': 'workcell-cpu', 'fetch': 'workcell-gpu', 'gpu': 'workcell-gpu', 'estop': 'workcell-gpu', 'recgen': 'recgen',
              'recgen-generate': 'recgen', 'sam3d': 'sam3d'}[STAGE]
SAM3D_WEIGHTS = 'panoptes-onprem-sam3d'
L4_RATE = .000222 + 4 * .0000131 + 16 * .00000222


def build_context() -> Path:
    """SRC/workcell = this repository, SRC/serving = panoptes-serving, code only + .dockerignore: scripts/onprem/stage_context.sh,
    the same context a customer builds from."""
    if not SERVING or not Path(SERVING, 'ehs_spatial').is_dir():
        raise SystemExit(f'set PANOPTES_SERVING to the panoptes-serving checkout (got {SERVING!r})')
    ctx = Path(tempfile.mkdtemp(prefix='panoptes-onprem-context-'))  # one per run: parallel runs must not share it
    subprocess.run(['bash', str(REPO / 'scripts/onprem/stage_context.sh'), str(ctx), SERVING], check=True, capture_output=True)
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
REPLAY = '''import shutil, sys, uuid
from pathlib import Path
sys.path.insert(0, '/serving')
from modal_apps.lucida_assets import generate_object, read_outputs
job = uuid.uuid4().hex; (Path('/cache/jobs') / job).mkdir(parents=True); shutil.copy(sys.argv[1], Path('/cache/jobs') / job / 'input.npz')
out = Path(sys.argv[2]); out.mkdir(parents=True)
for name, data in read_outputs(generate_object.remote(job, object_id=sys.argv[3], seed=int(sys.argv[4]))).items():
    (out / name).write_bytes(data)
'''  # run through run_stage.py as a driver script: generate_object on exactly the original payload bytes
RECGEN_VIEW_FILES = ('canonical_rgb_path', 'canonical_mask_path', 'pointmap_path', 'content_valid_path', 'conf_path', 'K_path', 'c2w_path')


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL')}  # no Modal identity reaches the stage
    # Modal's runtime puts its own client (/pkg/modal) on PYTHONPATH: a customer container has no such entry
    env['PYTHONPATH'] = os.pathsep.join(p for p in env.pop('PYTHONPATH', '').split(os.pathsep) if p and not p.startswith('/pkg'))
    if not env['PYTHONPATH']:
        del env['PYTHONPATH']
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


NO_MODAL = ("import importlib.util, os, sys; spec = importlib.util.find_spec('modal'); "
            "print(sys.executable, 'modal client:', spec.origin if spec else 'absent', '| PYTHONPATH', os.environ.get('PYTHONPATH')); "
            "sys.exit(1 if spec else 0)")


def stub_imports(py: str, apps: list, cwd='/workcell') -> dict:
    """Each app imported as run_stage.py imports it (scripts/onprem/modal_stub as `modal`): functions and entrypoints."""
    code = ("import json, sys; from pathlib import Path; sys.path.insert(0, '/workcell/scripts/onprem'); import run_stage\n"
            "for a in sys.argv[1:]:\n    m = run_stage.load_app(Path(a)); app = next(v for v in vars(m).values() if type(v).__name__ == 'App')\n"
            "    print(json.dumps([a, sorted(app.registered_functions), sorted(app.registered_entrypoints), sys.modules['modal'].__file__]))")
    return sh(f'stage apps import under the stub ({py})', [py, '-c', code, *apps], cwd=cwd)


AUDIT = r'''import json, os, runpy, socket, sys
events = []
def hook(event, args):
    if event == 'socket.connect':
        events.append(['connect', str(getattr(args[0], 'family', '')), repr(args[1])[:200]])
    elif event in ('socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyname_ex', 'socket.gethostbyaddr'):
        events.append([event.split('.', 1)[1], repr(args[0])[:200]])
    elif event in ('socket.sendto', 'socket.sendmsg'):
        events.append([event.split('.', 1)[1], repr(args[1:])[:200]])
    elif event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
        events.append([event, repr(args[:2])[:300]])
sys.addaudithook(hook)
import atexit
atexit.register(lambda: open(os.environ['AUDIT_OUT'], 'w').write(json.dumps(events)))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''  # python AUDIT.py SCRIPT ARGS...: every socket / DNS / subprocess call this Python process makes, written at exit


def outbound(events: list) -> list:
    """Audit events that leave the machine: an inet connect / sendto to a non-loopback address, or a DNS lookup of a host name."""
    import ast
    import ipaddress
    def local(host):
        if host in ('', 'localhost', None):
            return True
        try:
            return ipaddress.ip_address(str(host).strip('[]').split('%')[0]).is_loopback
        except ValueError:
            return False
    out = []
    for e in events:
        if e[0] == 'connect' and 'AF_UNIX' not in e[1]:
            try:
                host = ast.literal_eval(e[2])[0]
            except (ValueError, SyntaxError, IndexError, TypeError):
                host = e[2]
            if not local(host):
                out.append(e)
        elif e[0] in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr'):
            try:
                host = ast.literal_eval(e[1])
            except (ValueError, SyntaxError):
                host = e[1]
            host = host.decode() if isinstance(host, bytes) else host
            if not local(host):
                out.append(e)
        elif e[0] in ('sendto', 'sendmsg'):
            out.append(e)
    return out


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
    @app.function(image=image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0, block_network=True,
                  volumes={'/weights': modal.Volume.from_name(WEIGHTS)})  # only to return clearance B's 32 MB results.json
    def cpu_proof(inputs: dict) -> dict:
        start = time.monotonic()
        root = write_inputs(inputs)
        py, asm = '/opt/checks/bin/python', '/opt/assemble/bin/python'
        stub = clean_env(PYTHONPATH='/workcell/scripts/onprem/modal_stub')  # a module run directly (not via run_stage) imports the stub too
        steps = [sh('network probe', [py, '-c', PROBE]),
                 sh('no modal client (/opt/checks)', [py, '-c', NO_MODAL]), sh('no modal client (/opt/assemble)', [asm, '-c', NO_MODAL]),
                 stub_imports(py, ['modal_apps/workcell_view_checks.py', 'modal_apps/workcell_shape_check.py', 'modal_apps/workcell_clearance_b.py']),
                 stub_imports(asm, ['/serving/modal_apps/assemble_scene.py']),
                 sh('run_stage self-test', [py, 'scripts/onprem/run_stage.py', '--self-test']),
                 sh('fetch_weights self-test', [py, 'scripts/onprem/fetch_weights.py', '--self-test']),
                 sh('orthonormalise_cameras self-test', [py, 'scripts/onprem/orthonormalise_cameras.py', '--self-test']),
                 sh('airgap.sh self-test', ['bash', 'scripts/onprem/airgap.sh', 'self-test', '/tmp']),
                 sh('clearance_b self-test (/check/clearance_b.py)', [py, '/check/clearance_b.py']),
                 sh('layer build self-test', [py, 'scripts/workcell_layer_build.py', '--self-test']),
                 sh('configs in the image', [py, '-c', "import hashlib, json, pathlib; print(json.dumps({str(p.relative_to('/workcell')): "
                     "hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path('/workcell/configs').rglob('*.json'))}))"]),
                 sh('shape core self-test', [py, '/check/shape_core.py']),
                 sh('check modules self-tests', [py, '-c', "import sys; sys.path.insert(0, '/check'); import importlib\n"
                     "for n in ('floor', 'lines', 'plane_stereo', 'transfer'):\n    importlib.import_module('workcell_checks.' + n)._check(); print(n, 'ok')"]),
                 sh('moge check self-test (CPU part)', [py, 'modal_apps/workcell_moge_check.py'], env=stub),
                 sh('assemble venv imports', [asm, '-c', "import sys; sys.path.insert(0, '/serving/scripts/research'); import assemble_lucida_scene, "
                     "numpy, open3d, trimesh, cv2, scipy, PIL; print('assemble_lucida_scene', numpy.__version__, open3d.__version__, trimesh.__version__)"]),
                 sh('floor_masks self-test (assemble venv)', [asm, 'scripts/onprem/floor_masks.py', '--self-test']),
                 sh('floor check offline', [py, 'scripts/onprem/run_stage.py', '--offline', root / 'bundle', 'modal_apps/workcell_view_checks.py',
                                            '--checks', 'floor', *check_args(root, inputs, Path('/tmp/out/floor'))])]
        if inputs.get('targets'):  # clearance B (scripts/workcell_clearance_b.py as /check/clearance_b.py) on the same bundle, offline
            unpack(inputs['run'], Path('/tmp/in'))
            steps.append(sh('clearance B offline', [py, 'scripts/onprem/run_stage.py', '--offline', root / 'bundle', 'modal_apps/workcell_clearance_b.py',
                                                    *check_args(root, inputs, Path('/tmp/out/clearb'))[:-2], '--run-dir', '/tmp/in/run',
                                                    '--targets', inputs['targets'], '--out', '/tmp/out/clearb']))
        freeze = dict(checks=subprocess.run([py, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True).stdout,
                      assemble=subprocess.run(['uv', 'pip', 'freeze', '--python', asm], capture_output=True, text=True).stdout)
        result, spend = Path('/tmp/out/floor/results.json'), Path('/tmp/out/floor/spend-ledger.json')
        key = None
        if Path('/tmp/out/clearb').exists():  # a return value above Modal's inline size would be uploaded over the blocked network
            key = f'proof/{os.urandom(8).hex()}.tar.gz'
            (Path('/weights') / key).parent.mkdir(exist_ok=True)
            (Path('/weights') / key).write_bytes(pack(Path('/tmp/out/clearb'), 'clearb'))
            modal.Volume.from_name(WEIGHTS).commit()
        return dict(steps=steps, floor=json.loads(result.read_text()) if result.exists() else None, freeze=freeze,
                    floorStageLedger=json.loads(spend.read_text()) if spend.exists() else None, volumeKey=key,
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
    def gpu_proof(key: str) -> dict:
        start = time.monotonic()
        import pickle
        inputs = pickle.loads((Path('/weights') / key).read_bytes())  # inputs come through the volume: a large call argument is a
        (Path('/weights') / key).unlink()                              # blob download, which the blocked network may refuse
        want = set(inputs.get('steps', 'pi3x,moge,sam3').split(','))
        root = write_inputs(inputs)
        unpack(inputs['run'], Path('/tmp'))  # /tmp/run030: manifest + canonical frames (fresh Pi3X) and the original geometry
        fresh = Path('/tmp/pi3x-run'); (fresh / 'evidence').mkdir(parents=True)
        shutil.copy(Path('/tmp/run030/manifest.json'), fresh / 'manifest.json')
        shutil.copytree(Path('/tmp/run030/evidence/canonical'), fresh / 'evidence/canonical')
        run_stage = 'scripts/onprem/run_stage.py'
        steps = [sh('network probe', ['/opt/pi3x/bin/python', '-c', PROBE]),
                 sh('nvidia-smi', ['nvidia-smi', '-L']),
                 *[sh(f'no modal client ({v})', [f'/opt/{v}/bin/python', '-c', NO_MODAL]) for v in ('pi3x', 'sam3', 'moge')],
                 stub_imports('/opt/pi3x/bin/python', ['/serving/modal_apps/pi3x_geometry.py']),
                 stub_imports('/opt/moge/bin/python', ['modal_apps/workcell_moge_check.py']),
                 stub_imports('/opt/sam3/bin/python', ['modal_apps/workcell_mask_transfer.py', 'modal_apps/workcell_estop_mask.py',
                                                       'modal_apps/workcell_part_masks.py'])]
        if 'pi3x' in want:
            steps.append(sh('pi3x offline', ['/opt/pi3x/bin/python', run_stage, '--weights', '/weights', '/serving/modal_apps/pi3x_geometry.py', '--run', fresh]))
        if 'moge' in want:
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
        if 'sam3' in want:
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


if STAGE == 'estop':
    @app.function(image=image, gpu='L4', cpu=4, memory=16 * 1024, timeout=1800, retries=0, min_containers=0, block_network=True,
                  volumes={'/weights': modal.Volume.from_name(WEIGHTS)})
    def estop_proof(photos: dict, params: dict) -> dict:
        """The e-stop SAM 3 masks inside the GPU image: the app's own entrypoint, in-process, weights from the volume."""
        start = time.monotonic()
        src = Path('/tmp/in/photos'); src.mkdir(parents=True)
        for name, data in photos.items():
            (src / name).write_bytes(data)
        py, out = '/opt/sam3/bin/python', Path('/tmp/out/estop')
        steps = [sh('network probe', [py, '-c', PROBE]), sh('nvidia-smi', ['nvidia-smi', '-L']),
                 sh('e-stop SAM 3 masks offline', [py, 'scripts/onprem/run_stage.py', '--weights', '/weights', 'modal_apps/workcell_estop_mask.py',
                                                  '--photos-dir', src, '--photo', ','.join(photos), '--out', out,
                                                  *[a for k, v in params.items() for a in (f'--{k}', str(v))]])]
        key = f'proof/{os.urandom(8).hex()}.tar.gz'  # masks return through the volume (large returns need the network)
        (Path('/weights') / key).parent.mkdir(exist_ok=True)
        (Path('/weights') / key).write_bytes(pack(out, 'estop') if out.exists() else b'')
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


if STAGE == 'recgen-generate':
    @app.function(image=image, cpu=4, memory=8 * 1024, timeout=1800, retries=0, min_containers=0,
                  volumes={'/weights': modal.Volume.from_name(WEIGHTS)})
    def recgen_fetch() -> dict:
        """fetch_weights.py --models recgen with the image's own HF_HUB_OFFLINE=1 (the script must switch it off itself)."""
        start = time.monotonic()
        fw = '/workcell/scripts/onprem/fetch_weights.py'
        steps = [sh('fetch recgen (HF_HUB_OFFLINE=1 in env)', ['python', fw, '--cache', '/weights', '--models', 'recgen'], cwd='/serving'),
                 sh('verify (offline re-hash)', ['python', fw, '--cache', '/weights', '--verify'], cwd='/serving')]
        modal.Volume.from_name(WEIGHTS).commit()
        stage = Path('/weights/recgen/recgen-weights-manifest.json')
        import hashlib
        return dict(steps=steps, recgen=json.loads(Path('/weights/manifest.json').read_text())['models'].get('recgen'),
                    stageManifestSha256=hashlib.sha256(stage.read_bytes()).hexdigest() if stage.exists() else None,
                    containerSeconds=time.monotonic() - start)

    @app.function(image=image, gpu='A100-80GB', cpu=8, memory=64 * 1024, timeout=1800, retries=0, min_containers=0, block_network=True,
                  volumes={'/weights': modal.Volume.from_name(WEIGHTS)})
    def recgen_generate(run: bytes, object_id: str, seed: int, payload: bytes = b'') -> dict:
        """The serving driver unchanged, in this process: /cache -> /weights/recgen (run_stage --weights), network blocked.
        With payload (the original run's input.npz), generate_object also runs once more on exactly those bytes (replay)."""
        start = time.monotonic()
        unpack(run, Path('/tmp'))
        rs = ['python', '/workcell/scripts/onprem/run_stage.py', '--weights', '/weights']
        listing = lambda: sorted(p.name for p in Path('/weights/recgen').iterdir())
        before = listing()
        steps = [sh('network probe', ['python', '-c', PROBE], cwd='/serving'),
                 sh('nvidia-smi', ['nvidia-smi', '-L'], cwd='/serving'),
                 sh('no modal client', ['python', '-c', NO_MODAL], cwd='/serving'),
                 sh('recgen environment check', [*rs, '/serving/modal_apps/lucida_assets.py', '--mode', 'check',
                                                 '--output-dir', '/tmp/run/generation/environment-recgen'], cwd='/serving'),
                 sh(f'generate {object_id}', [*rs, '/serving/scripts/research/generate_lucida_assets.py', '--root', '/tmp/run',
                                              '--object-ids', object_id, '--seed', str(seed)], cwd='/serving')]
        if payload:
            Path('/tmp/replay-input.npz').write_bytes(payload)
            Path('/tmp/replay.py').write_text(REPLAY)
            steps.append(sh(f'replay original input.npz of {object_id}', [*rs, '/tmp/replay.py', '/tmp/replay-input.npz',
                                                                         '/tmp/run/generation/replay', object_id, str(seed)], cwd='/serving'))
        import glob
        weights_check = dict(recgenEntriesBefore=before, recgenEntriesAfter=listing(),
                             jobCopiesInWeights=sorted(str(p) for p in Path('/weights/recgen').rglob('input.npz')),
                             cacheLinkLeft=os.path.lexists('/cache'), overlaysLeft=glob.glob(os.path.join(tempfile.gettempdir(), 'onprem-run-*')))
        key = f'proof/{os.urandom(8).hex()}.tar.gz'  # outputs return through the volume (large returns need the network)
        (Path('/weights') / key).parent.mkdir(exist_ok=True)
        (Path('/weights') / key).write_bytes(pack(Path('/tmp/run/generation'), 'generation'))
        modal.Volume.from_name(WEIGHTS).commit()
        return dict(steps=steps, volumeKey=key, weightsCheck=weights_check, containerSeconds=time.monotonic() - start)


SAM3D_DRIVER = r'''import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workcell/modal_apps')
import sam3d_research as s
inp, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
model, rec = s.SAM3DObjects(), {}
for f in sorted(inp.glob('*.npz')):
    d = np.load(f); t = time.monotonic()
    r = model.run.remote(d['rgb'], d['mask'], d['pointmap'], 42)
    if 'error' in r:
        rec[f.stem] = dict(error=r['error'][-3000:])
    else:
        np.savez_compressed(out / f.name, vertices=r['vertices'], faces=r['faces'].astype(np.int32), colors=r['colors'],
                            object_to_camera_p3d=r['objectToCamera'])
        rec[f.stem] = dict(seconds=time.monotonic() - t, vertices=len(r['vertices']), gpu=r['gpu'], pins=r['pins'])
    print(f.stem, json.dumps(rec[f.stem])[:400], flush=True)
(out / 'record.json').write_text(json.dumps(rec, indent=1))
'''  # run by run_stage.py as a driver script: SAM3DObjects().run.remote runs here, after its @enter
HUB_PROBE = "import torch\nm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=False)\nprint(type(m).__name__)\n"

if STAGE == 'sam3d':
    @app.function(image=image, gpu='A100-80GB', cpu=4, memory=32 * 1024, timeout=1800, retries=0, min_containers=0,
                  volumes={'/weights': modal.Volume.from_name(SAM3D_WEIGHTS)})  # network ON on purpose: the audit must find nothing
    def sam3d_proof(inputs: dict) -> dict:
        start = time.monotonic()
        src, out = Path('/tmp/in'), Path('/tmp/out'); src.mkdir(parents=True)
        for name, data in inputs.items():
            (src / name).write_bytes(data)
        for name, text in (('audit.py', AUDIT), ('driver.py', SAM3D_DRIVER), ('hub_probe.py', HUB_PROBE)):
            Path('/tmp', name).write_text(text)
        rs = ['python', '/tmp/audit.py', '/workcell/scripts/onprem/run_stage.py']
        audited = lambda name, cmd, tag: (sh(name, cmd, env=clean_env(AUDIT_OUT=f'/tmp/audit-{tag}.json')), tag)
        runs = [audited('torch.hub.load without run_stage (baseline, network on)', ['python', '/tmp/audit.py', '/tmp/hub_probe.py'], 'hub-plain'),
                audited('torch.hub.load through run_stage (network on)', [*rs, '/tmp/hub_probe.py'], 'hub-pinned'),
                audited('sam3d generate through run_stage --weights (network on)', [*rs, '--weights', '/weights', '/tmp/driver.py', src, out], 'generate')]
        steps = [sh('network probe (network is on here)', ['python', '-c', PROBE]), sh('nvidia-smi', ['nvidia-smi', '-L']),
                 sh('no modal client', ['python', '-c', NO_MODAL]), stub_imports('python', ['modal_apps/sam3d_research.py'])]
        audit = {}
        for step, tag in runs:
            steps.append(step)
            ev = json.loads(Path(f'/tmp/audit-{tag}.json').read_text()) if Path(f'/tmp/audit-{tag}.json').exists() else None
            audit[tag] = dict(events=len(ev) if ev is not None else None, outbound=outbound(ev) if ev is not None else None,
                              subprocesses=[e for e in ev if e[0].startswith(('subprocess', 'os.'))][:40] if ev is not None else None,
                              unixConnects=sum(e[0] == 'connect' and 'AF_UNIX' in e[1] for e in ev) if ev is not None else None)
        files = {p.name: p.read_bytes() for p in out.glob('*')} if out.exists() else {}
        return dict(steps=steps, audit=audit, files=files,
                    checkpointDir=sorted(p.name for p in Path('/weights/sam3d/torch-hub/checkpoints').iterdir()),
                    containerSeconds=time.monotonic() - start)


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
            ref_path = run_dir / 'evidence/floor-masks' / f"{fr['frame_id']}.png"
            if not ref_path.exists():  # a run without frozen product floor masks (090): compared offline by the caller
                rows[fr['frame_id']] = dict(instances=len(masks), scores=view[0]['scores'], iouVsProductRunFloorMask=None)
                continue
            ref_mask = np.asarray(Image.open(ref_path)) > 0
            x0, y0, x1, y1 = fr['content_rect_xyxy']; inside = np.zeros_like(ref_mask); inside[y0:y1, x0:x1] = True
            a, b = canon & inside, ref_mask & inside
            rows[fr['frame_id']] = dict(instances=len(masks), scores=view[0]['scores'], iouVsProductRunFloorMask=float((a & b).sum() / max(1, (a | b).sum())),
                                        newPixels=int(a.sum()), productRunPixels=int(b.sum()))
            Image.fromarray(canon.astype(np.uint8) * 255).save(out / f"sam3-floor-{fr['frame_id']}.png")
        cmp['sam3Floor'] = rows
    return cmp


def compare_masks(new: Path, ref: Path) -> dict:
    """Every candidate PNG of a workcell_estop_mask.py output against another run's: bytes, pixels, IoU; instance scores."""
    import hashlib
    import numpy as np
    from PIL import Image
    a, b = json.loads((new / 'results.json').read_text()), json.loads((ref / 'results.json').read_text())
    out = dict(params={k: (a.get(k), b.get(k)) for k in ('text', 'tile', 'threshold', 'keep')}, photos={})
    for name, pa in a['photos'].items():
        pb = b['photos'][name]; rows = []
        for ia, ib in zip([i for i in pa['instances'] if 'mask' in i], [i for i in pb['instances'] if 'mask' in i]):
            ma, mb = np.asarray(Image.open(new / ia['mask'])) > 0, np.asarray(Image.open(ref / ib['mask'])) > 0
            rows.append(dict(mask=ia['mask'], sameFile=ia['mask'] == ib['mask'], prompt=(ia['prompt'], ib['prompt']), scoreDiff=ia['score'] - ib['score'],
                             bbox=(ia['bbox'], ib['bbox']), bytesEqual=hashlib.sha256((new / ia['mask']).read_bytes()).digest() == hashlib.sha256((ref / ib['mask']).read_bytes()).digest(),
                             pixelsEqual=bool(np.array_equal(ma, mb)), iou=float((ma & mb).sum() / max(1, (ma | mb).sum()))))
        out['photos'][name] = dict(instances=(len(pa['instances']), len(pb['instances'])), candidates=rows,
                                   allBytesEqual=all(r['bytesEqual'] for r in rows) and len(rows) == sum('mask' in i for i in pb['instances']),
                                   minIoU=min((r['iou'] for r in rows), default=None), maxAbsScoreDiff=max((abs(r['scoreDiff']) for r in rows), default=None))
    return out


def recgen_run(run: Path, object_id: str) -> bytes:
    """The run's manifest, objects.json and the object's view files only (what payload_for_object reads), plus a
    generation/gpu-budget.json already at its 3600 s Modal limit: the Modal gate would refuse the call, on-prem must not."""
    obj = next(o for o in json.loads((run / 'evidence/objects.json').read_text())['objects'] if o['object_id'] == object_id)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / 'run'
        for rel in ['manifest.json', 'evidence/objects.json', *[v[k] for v in obj['views'] for k in RECGEN_VIEW_FILES]]:
            (stage / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(run / rel, stage / rel)
        (stage / 'generation').mkdir()
        (stage / 'generation/gpu-budget.json').write_text(json.dumps(dict(
            limit_seconds=3600, timeout_per_call_seconds=600, accounting='proof: pre-filled to the limit',
            calls=[dict(object_id='proof-budget-filler', charged_seconds=3600, status='complete')]), indent=2) + '\n')
        return pack(stage, 'run')


def npz_equal(a: Path, b: Path) -> bool:
    import numpy as np
    x, y = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    return sorted(x.files) == sorted(y.files) and all(x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k]) for k in x.files)


def compare_meshes(new_dir: Path, old_dir: Path) -> dict:
    """object.ply / posed-object.ply: counts, SHA-256, symmetric Hausdorff and Chamfer (vertex nearest neighbours, native units)."""
    import hashlib
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    out = {}
    for name in ('object.ply', 'posed-object.ply'):
        a, b = trimesh.load(new_dir / name, process=False), trimesh.load(old_dir / name, process=False)
        row = dict(vertices=(len(a.vertices), len(b.vertices)), faces=(len(a.faces), len(b.faces)),
                   sha256Equal=hashlib.sha256((new_dir / name).read_bytes()).digest() == hashlib.sha256((old_dir / name).read_bytes()).digest())
        if row['vertices'][0] == row['vertices'][1]:
            row['sameVertexArrays'] = bool(np.array_equal(a.vertices, b.vertices)); row['sameFaces'] = bool(np.array_equal(a.faces, b.faces))
            row['vertexMaxAbsDiffNative'] = float(np.abs(np.asarray(a.vertices) - np.asarray(b.vertices)).max())
        ab, ba = cKDTree(b.vertices).query(a.vertices)[0], cKDTree(a.vertices).query(b.vertices)[0]
        row.update(hausdorffNative=float(max(ab.max(), ba.max())), chamferMeanNative=float((ab.mean() + ba.mean()) / 2),
                   bboxDiagonalNative=float(np.linalg.norm(np.ptp(np.asarray(b.vertices), axis=0))))
        out[name] = row
    return out


def compare_sam3d(new: Path, old: Path) -> dict:
    """Same input, same seed: arrays, pose, and vertex nearest-neighbour distances in the pointmap's camera (native units)."""
    import numpy as np
    from scipy.spatial import cKDTree
    a, b = np.load(new), np.load(old)
    row = dict(vertices=(len(a['vertices']), len(b['vertices'])), poseMaxAbsDiff=float(np.abs(a['object_to_camera_p3d'] - b['object_to_camera_p3d']).max()),
               identicalArrays={k: bool(a[k].shape == b[k].shape and np.array_equal(a[k], b[k])) for k in ('vertices', 'faces', 'colors', 'object_to_camera_p3d')})
    Va, Vb = [V @ M[:3, :3].T + M[:3, 3] for V, M in ((a['vertices'], a['object_to_camera_p3d']), (b['vertices'], b['object_to_camera_p3d']))]
    ab, ba = cKDTree(Vb).query(Va)[0], cKDTree(Va).query(Vb)[0]
    row.update(cameraHausdorff=float(max(ab.max(), ba.max())), cameraChamferMean=float((ab.mean() + ba.mean()) / 2))
    return row


@app.local_entrypoint()
def main(out: str, view: str = '', photos_dir: str = '', photo: str = '', layer_url: str = '', api: str = '', bundle: str = '',
         reference: str = '', run_dir: str = '', moge_reference: str = '', models: str = 'pi3x,moge3,sam3',
         sam3_source: str = '/v/src-sam3/huggingface/hub', object_id: str = 'left_post', seed: int = 42, steps: str = 'pi3x,moge,sam3',
         targets: str = '', clearance_reference: str = '', replay: bool = False, fetch: bool = True, inputs: str = ''):
    dest = Path(out)
    if dest.exists():
        raise ValueError('Choose a fresh output directory')
    dest.mkdir(parents=True)
    start = time.monotonic()
    npz_dir = inputs
    if STAGE in ('cpu', 'gpu'):
        names = dict(item.split('=', 1) for item in photo.split(','))
        inputs = dict(view=Path(view).read_bytes(), photos={n: (Path(photos_dir) / n).read_bytes() for n in names.values()}, photo=photo,
                      layer_url=layer_url, api=api, bundle=bundle_tar(bundle), steps=steps)
    if STAGE == 'cpu':
        if targets:  # clearance B's Pi3X inputs: the run's per-frame arrays (as modal_apps/workcell_clearance_b.py reads them)
            run = Path(run_dir).resolve()
            with tempfile.TemporaryDirectory() as tmp:
                for f in sorted((run / 'geometry/frames').iterdir()):
                    (Path(tmp) / 'run/geometry/frames' / f.name).mkdir(parents=True)
                    for name in ('pts3d.npy', 'conf.npy', 'valid_mask.npy', 'content_valid_mask.npy'):
                        shutil.copy(f / name, Path(tmp) / 'run/geometry/frames' / f.name / name)
                inputs.update(run=pack(Path(tmp) / 'run', 'run'), targets=targets)
        result = cpu_proof.remote(inputs)
        if result['floor'] and reference:
            result['comparison'] = floor_diff(json.loads(Path(reference).read_text()), result['floor'])
        if result.get('volumeKey'):
            volume = modal.Volume.from_name(WEIGHTS)
            unpack(b''.join(volume.read_file(result['volumeKey'])), dest)
            volume.remove_file(result['volumeKey'])
            if clearance_reference:
                ref_dir, new_dir = Path(clearance_reference).parent, dest / 'clearb'
                strip = lambda r: {k: v for k, v in r.items() if k != 'containerSeconds'}
                result['clearanceComparison'] = dict(
                    identicalExceptSeconds=strip(json.loads(Path(clearance_reference).read_text())) == strip(json.loads((new_dir / 'results.json').read_text())),
                    overlaysByteEqual={p.name: (ref_dir / p.name).exists() and (ref_dir / p.name).read_bytes() == p.read_bytes() for p in sorted(new_dir.glob('*.jpg'))})
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
        import pickle
        volume = modal.Volume.from_name(WEIGHTS)
        key = f'proof/in-{os.urandom(8).hex()}.pkl'
        with volume.batch_upload() as up:
            up.put_file(io.BytesIO(pickle.dumps(inputs)), '/' + key)
        result = gpu_proof.remote(key)
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
        if 'sam3' in result:  # full-resolution RLE per instance: the input of scripts/onprem/floor_masks.py
            (dest / 'sam3.json').write_text(json.dumps(result.pop('sam3')))
        result['ledger'] = ledger(dest, result['containerSeconds'], GPU_RATE, 'A100-80GB, 8 CPU, 32 GiB, block_network', time.monotonic() - start)
    elif STAGE == 'estop':
        ref = json.loads((Path(reference) / 'results.json').read_text()) if reference else {}
        params = {k: ref[k] for k in ('text', 'tile', 'threshold', 'keep') if k in ref}
        names = photo.split(',')
        result = estop_proof.remote({n: (Path(photos_dir) / n).read_bytes() for n in names}, params)
        volume = modal.Volume.from_name(WEIGHTS)
        blob = b''.join(volume.read_file(result['volumeKey']))
        volume.remove_file(result['volumeKey'])
        if blob:
            unpack(blob, dest)
            if reference:
                result['comparison'] = compare_masks(dest / 'estop', Path(reference))
        result['ledger'] = ledger(dest, result['containerSeconds'], L4_RATE, '1 L4, 4 CPU, 16 GiB, block_network', time.monotonic() - start)
    elif STAGE == 'recgen-generate':
        run = Path(run_dir).resolve()
        fetched = recgen_fetch.remote() if fetch else dict(steps=[], containerSeconds=0.)
        fetch_call = time.monotonic() - start
        generated = recgen_generate.remote(recgen_run(run, object_id), object_id, seed,
                                           (run / 'generation' / object_id / 'input.npz').read_bytes() if replay else b'')
        volume = modal.Volume.from_name(WEIGHTS)
        unpack(b''.join(volume.read_file(generated['volumeKey'])), dest)
        volume.remove_file(generated['volumeKey'])
        new, old = dest / 'generation' / object_id, run / 'generation' / object_id
        result = dict(fetch=fetched, steps=fetched['steps'] + generated['steps'], weightsCheck=generated['weightsCheck'],
                      containerSeconds=fetched['containerSeconds'] + generated['containerSeconds'])
        budget = dest / 'generation/gpu-budget.json'
        if budget.exists():
            calls = json.loads(budget.read_text())['calls']
            result['budgetGate'] = dict(limitSeconds=3600, chargedBeforeSeconds=3600, calls=calls,
                                        onPremCallRan=any(c.get('onprem') and c['object_id'] == object_id and c['status'] == 'complete' for c in calls))
        if (new / 'object.ply').exists():
            rec, ref = json.loads((new / 'output.json').read_text()), json.loads((old / 'output.json').read_text())
            same = lambda k: (rec.get(k), ref.get(k))
            result['comparison'] = dict(object=object_id, seed=seed,
                                      inputArraysEqual=npz_equal(new / 'input.npz', old / 'input.npz'),  # zip member times differ, arrays must not
                                      **{k: same(k) for k in ('status', 'source_payload_sha256', 'weights_manifest_sha256', 'used_weight_files', 'hardware',
                                                              'vertices', 'faces', 'watertight', 'body_count', 'output_sha256', 'inference_seconds')},
                                      poseMatrixMaxAbsDiff=float(max(abs(x - y) for r1, r2 in zip(rec['pose_matrix'], ref['pose_matrix']) for x, y in zip(r1, r2))),
                                      meshes=compare_meshes(new, old))
        replay = dest / 'generation/replay'
        if (replay / 'object.ply').exists():
            rp, orig = json.loads((replay / 'output.json').read_text()), json.loads((old / 'output.json').read_text())
            result['replayOriginalInput'] = dict(payloadSha256=(rp['source_payload_sha256'], orig['source_payload_sha256']), status=rp['status'],
                                                 outputSha256Equal={k: rp['output_sha256'].get(k) == v for k, v in orig['output_sha256'].items()},
                                                 poseMatrixMaxAbsDiff=float(max(abs(x - y) for r1, r2 in zip(rp['pose_matrix'], orig['pose_matrix']) for x, y in zip(r1, r2))),
                                                 meshes=compare_meshes(replay, old))
        result['ledger'] = ledger(dest, generated['containerSeconds'], GPU_RATE + 32 * .00000222, 'A100-80GB, 8 CPU, 64 GiB, block_network (generate)',
                                  time.monotonic() - start - fetch_call)
        result['ledger']['fetch'] = dict(functionSeconds=fetched['containerSeconds'], callSeconds=fetch_call,
                                         estimateUsd=FETCH_RATE * fetched['containerSeconds'], hardware='4 CPU, 8 GiB, network on (fetch only)')
        result['ledger']['estimateUsdTotal'] = result['ledger']['estimateUsd'] + result['ledger']['fetch']['estimateUsd']
        (dest / 'spend-ledger.json').write_text(json.dumps(result['ledger'], indent=2) + '\n')
    elif STAGE == 'sam3d':
        blobs = {p.name: p.read_bytes() for p in sorted(Path(npz_dir).glob('*.npz'))}
        result = sam3d_proof.remote(blobs)
        for name, data in result.pop('files').items():
            (dest / 'out').mkdir(exist_ok=True); (dest / 'out' / name).write_bytes(data)
        if reference:
            result['comparison'] = {p.stem: compare_sam3d(dest / 'out' / p.name, Path(reference) / p.name)
                                    for p in sorted(Path(npz_dir).glob('*.npz')) if (dest / 'out' / p.name).exists() and (Path(reference) / p.name).exists()}
        result['ledger'] = ledger(dest, result['containerSeconds'], .000694 + 4 * .0000131 + 32 * .00000222, 'A100-80GB, 4 CPU, 32 GiB, network on',
                                  time.monotonic() - start)
    else:
        result = recgen_proof.remote()
        result['ledger'] = ledger(dest, result['containerSeconds'], 2 * .0000131 + 8 * .00000222, '2 CPU, 8 GiB, block_network', time.monotonic() - start)
    (dest / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + '\n')
    for s in result.get('steps', []):
        print(f"[{'ok' if s['rc'] == 0 else 'FAIL rc=' + str(s['rc'])}] {s['step']} {s['seconds']}s :: {(s['stdout'].strip().splitlines() or [''])[-1][:300]}")
        if s['rc']:
            print('   stderr:', s['stderr'][-1500:])
    print(json.dumps({k: result[k] for k in ('comparison', 'ledger') if k in result}, default=str)[:4000])
