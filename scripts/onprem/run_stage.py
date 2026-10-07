"""Run any Modal app (or driver script) of the workcell photo pipeline in this process, without Modal: the on-prem runner.

  python scripts/onprem/run_stage.py [--offline BUNDLE | --record BUNDLE] [--weights DIR] APP.py[::ENTRYPOINT] [flags as for `modal run`]
  python scripts/onprem/run_stage.py [...] SCRIPT.py [its own arguments]          (a file without a Modal local entrypoint)
  python scripts/onprem/run_stage.py --self-test
  python scripts/onprem/run_stage.py geometry --run RUN --start GEOM --output NEW (--weights DIR | --roma DIR)   (geometry())

The module is imported as `modal run` imports it, but `import modal` is scripts/onprem/modal_stub/modal.py (put first on sys.path
and PYTHONPATH): the images carry no modal client, and no account, token, config or network is ever used. Then:
- the app's local entrypoint runs unchanged, and every Function.remote / spawn / map / starmap runs the function body here
  (class methods after their @enter);
- an image's add_local_file / add_local_dir container paths (`REPO / 'x'` -> '/check/y') resolve to the repository files
  through an import hook; images built from docker/ already hold those paths, then nothing is mapped;
- a Volume is the local directory at its container mount path: commit / reload are no-ops, read_file and batch_upload use it;
- --record BUNDLE saves every urllib response the stage reads (index.json + blobs/<sha256>); --offline BUNDLE replays them,
  refuses every other URL and every non-loopback socket / DNS lookup in this process, and sets HF_HUB_OFFLINE=1;
- --weights DIR (written by fetch_weights.py) gives each model's container path (/cache, /tmp/pi3x, ...) a run-private overlay
  in a scratch directory (TMPDIR): one symlink per weight entry, so whatever a stage writes there (RecGen's /cache/jobs/<id>,
  which holds photo data) lands in scratch, never in DIR, and is deleted when the stage ends; sets HF_HOME and HF_HUB_OFFLINE=1;
- torch.hub.load(..., source='github') is served from the vendored copy in torch.hub's own directory (<hub>/<owner>_<repo>_<ref>)
  with source='local', and torch.hub downloads other than file:// are refused: no GitHub probe even when the container has a
  network;
- PANOPTES_ONPREM=1 is set (stages switch off Modal-only rules, e.g. RecGen's Modal GPU budget gate);
- every *spend-ledger.json a stage writes (Path.write_text) is recorded as mode 'on-prem in-process' with this machine's
  hardware (cgroup CPU quota and memory limit, as docker --cpus / --memory set them) and no USD estimate.
  The stage's own stdout line still shows its Modal list-rate dict.
"""
from __future__ import annotations

import argparse
import ast
import atexit
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import io
import ipaddress
from itertools import combinations
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import urllib.response

STUB = Path(__file__).resolve().parent / 'modal_stub'
LOOPBACK_NAMES = ('', 'localhost')  # '' = getaddrinfo(None, ...): a local (passive) address


def use_stub():
    """`import modal` -> the on-prem stub, here and in child processes."""
    loaded = sys.modules.get('modal')
    if loaded is not None and not getattr(loaded, 'ONPREM_STUB', False):
        raise RuntimeError('the real modal client is already imported in this process; run_stage.py needs the on-prem stub')
    if str(STUB) not in sys.path:
        sys.path.insert(0, str(STUB))
    paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    if str(STUB) not in paths:
        os.environ['PYTHONPATH'] = os.pathsep.join([str(STUB), *filter(None, paths)])
    import modal
    assert modal.ONPREM_STUB, modal.__file__
    return modal


# ---------------------------------------------------------------- network: record / offline replay
def _loopback(host) -> bool:
    """Exact names and parsed addresses only: 'localhost', '', 127.0.0.0/8, ::1 (also IPv4-mapped), 0.0.0.0 / ::."""
    host = host.decode() if isinstance(host, bytes) else str(host or '')
    if host in LOOPBACK_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(host.strip('[]').split('%')[0])
    except ValueError:
        return False
    ip = getattr(ip, 'ipv4_mapped', None) or ip
    return ip.is_loopback or ip.is_unspecified


def block_network():
    """Refuse non-loopback sockets and DNS in this process (subprocesses and C-level clients are not covered: use the
    container's own network isolation for those, as onprem_image_proof.py does with block_network=True)."""
    real_connect, real_connect_ex, real_gai = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def allowed(sock, address):
        return sock.family == getattr(socket, 'AF_UNIX', None) or _loopback(address[0] if isinstance(address, tuple) else address)

    def connect(sock, address):
        if not allowed(sock, address):
            raise OSError(f'onprem offline: network access to {address} refused')
        return real_connect(sock, address)

    def connect_ex(sock, address):
        if not allowed(sock, address):
            raise OSError(f'onprem offline: network access to {address} refused')
        return real_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if not _loopback(host):
            raise socket.gaierror(f'onprem offline: DNS lookup of {host} refused')
        return real_gai(host, *args, **kwargs)

    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo


def _response(data: bytes, url: str, headers: dict | None = None):
    from email.message import Message
    msg = Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.response.addinfourl(io.BytesIO(data), msg, url, 200)


def record_urls(bundle: Path):
    """Wrap urllib.request.urlopen: each response body is kept under bundle/blobs/<sha256>, index.json maps url -> blob."""
    real = urllib.request.urlopen
    (bundle / 'blobs').mkdir(parents=True, exist_ok=True)
    index_path = bundle / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    def urlopen(url, *args, **kwargs):
        key = url.full_url if isinstance(url, urllib.request.Request) else str(url)
        with real(url, *args, **kwargs) as r:
            data, ctype = r.read(), r.headers.get('Content-Type')
        digest = hashlib.sha256(data).hexdigest()
        (bundle / 'blobs' / digest).write_bytes(data)
        index[key] = dict(sha256=digest, bytes=len(data), contentType=ctype)
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
        return _response(data, key, {'Content-Type': ctype} if ctype else None)

    urllib.request.urlopen = urlopen


def replay_urls(bundle: Path):
    """urllib.request.urlopen served from a recorded bundle; any other URL fails, every blob is SHA-256 checked."""
    index = json.loads((bundle / 'index.json').read_text())
    served = set()

    def urlopen(url, *args, **kwargs):
        key = url.full_url if isinstance(url, urllib.request.Request) else str(url)
        if key not in index:
            raise urllib.error.URLError(f'onprem offline: {key} is not in the bundle {bundle}')
        data = (bundle / 'blobs' / index[key]['sha256']).read_bytes()
        if hashlib.sha256(data).hexdigest() != index[key]['sha256']:
            raise ValueError(f'bundle blob for {key} does not match its SHA-256')
        served.add(key)
        ctype = index[key].get('contentType')
        return _response(data, key, {'Content-Type': ctype} if ctype else None)

    urllib.request.urlopen = urlopen
    return served


# ---------------------------------------------------------------- weights (fetch_weights.py manifest): run-private overlays
OVERLAYS: list = []  # (container path, overlay dir) made by this process


def _entries(cache: Path, model: dict) -> set:
    """Top-level names under the model's directory that hold its manifest files (only verified weights are exposed)."""
    extras = model.get('extras', {})
    return {Path(extras[rel] if rel in extras else Path(model['snapshot'], rel)).relative_to(model['path']).parts[0]
            for rel in model['files']}


def link_weights(cache: Path, scratch: Path | None = None) -> dict:
    """Each model with a container path ('link') -> symlink to a fresh overlay directory scratch/<model> holding one symlink per
    weight entry. The stage reads the weights unchanged; new files it writes there stay in scratch (deleted by cleanup())."""
    manifest = json.loads((cache / 'manifest.json').read_text())
    os.environ.setdefault('HF_HOME', str(cache / manifest.get('hfHome', 'hf')))
    os.environ.setdefault('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))
    if not OVERLAYS:
        atexit.register(cleanup)  # also for callers other than main()
    done = {}
    for name, model in manifest['models'].items():
        if not model.get('link'):
            continue
        target, path = cache / model['path'], Path(model['link'])
        if path.is_symlink():
            current = Path(os.readlink(path))
            if current.exists() and current.parent.name.startswith('onprem-run-'):
                raise RuntimeError(f'onprem: {path} -> {current} belongs to another run_stage.py (or a crashed one: it may hold '
                                   f'job copies with photo data; delete {current.parent} and {path})')
            path.unlink()  # an older direct link to the weights, or a dangling one
        elif path.exists():  # a real directory a deployment mounted there itself: leave it
            print(f'onprem: {path} exists, not linking {name}', file=sys.stderr)
            continue
        scratch = scratch or Path(tempfile.mkdtemp(prefix='onprem-run-'))
        overlay = scratch / name
        overlay.mkdir(parents=True)
        for entry in sorted(_entries(cache, model)):
            (overlay / entry).symlink_to(target / entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(overlay, target_is_directory=True)
        OVERLAYS.append((path, overlay))
        done[name] = f'{path} -> {overlay} ({len(list(overlay.iterdir()))} entries of {target})'
    return done


def cleanup():
    """Remove this process's overlays (their symlinks into the weights are unlinked, never followed) and container links."""
    while OVERLAYS:
        path, overlay = OVERLAYS.pop()
        if path.is_symlink() and Path(os.readlink(path)) == overlay:
            path.unlink()
        shutil.rmtree(overlay, ignore_errors=True)
        try:
            overlay.parent.rmdir()  # the run's scratch directory, once its last overlay is gone
        except OSError:
            pass


# ---------------------------------------------------------------- torch.hub: vendored code only, no download
def _pin_hub(hub):
    if getattr(hub.load, '_onprem', False):
        return
    real = hub.load

    def load(repo_or_dir, *args, source='github', **kwargs):
        if str(source).lower() != 'github':
            return real(repo_or_dir, *args, source=source, **kwargs)
        repo, _, ref = str(repo_or_dir).partition(':')
        owner, _, name = repo.partition('/')
        for r in ([ref] if ref else ['main', 'master']):  # the directories torch.hub itself would use
            local = Path(hub.get_dir()) / f"{owner}_{name}_{r.replace('/', '_')}"
            if local.is_dir():
                return real(str(local), *args, source='local', **kwargs)
        raise RuntimeError(f'onprem: torch.hub repository {repo_or_dir} is not vendored under {hub.get_dir()}')

    real_download = hub.download_url_to_file

    def download(url, *args, **kwargs):  # file:// is a local copy (RecGen passes its checkpoint so); anything else is refused
        if not str(url).startswith('file://'):
            raise RuntimeError(f'onprem: torch.hub download of {url} refused: put the file in the weights directory '
                               f'(torch.hub checkpoints: {Path(hub.get_dir()) / "checkpoints"})')
        return real_download(url, *args, **kwargs)

    load._onprem = True
    hub.load, hub.download_url_to_file = load, download


class _HubPin(importlib.abc.MetaPathFinder):
    """Patches torch.hub right after it is imported (stages import torch late, inside function bodies)."""
    def find_spec(self, name, path=None, target=None):
        if name != 'torch.hub':
            return None
        for finder in sys.meta_path:
            spec = None if finder is self or not hasattr(finder, 'find_spec') else finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            _pin_hub(module)
        spec.loader.exec_module = exec_module
        return spec


def pin_torch_hub():
    if 'torch.hub' in sys.modules:
        _pin_hub(sys.modules['torch.hub'])
    elif not any(isinstance(f, _HubPin) for f in sys.meta_path):
        sys.meta_path.insert(0, _HubPin())


# ---------------------------------------------------------------- spend ledgers: no Modal call, no Modal charge
def cgroup_limits(root=Path('/sys/fs/cgroup'), proc=Path('/proc/self/cgroup')):
    """(CPUs, memory bytes) of this process's cgroup and its parents - docker --cpus / --memory - or None where unlimited.
    cgroup v2 (cpu.max, memory.max) and v1 (cpu.cfs_quota_us / cfs_period_us, memory.limit_in_bytes)."""
    try:
        lines = [line.split(':', 2) for line in proc.read_text().splitlines()]
    except OSError:
        return None, None
    rel = {c: path.lstrip('/') for _, ctrls, path in lines for c in (ctrls.split(',') if ctrls else [''])}

    def walk(base, path, name):
        p = Path(path)
        for d in [p, *p.parents]:
            try:
                yield (base / d / name).read_text().split()
            except OSError:
                continue
    cpus, mem = [], []
    if '' in rel:
        cpus += [int(q) / int(p) for q, p, *_ in walk(root, rel[''], 'cpu.max') if q != 'max']
        mem += [int(m[0]) for m in walk(root, rel[''], 'memory.max') if m[0] != 'max']
    if 'cpu' in rel:
        quota = [int(q[0]) for q in walk(root / 'cpu', rel['cpu'], 'cpu.cfs_quota_us')]
        period = [int(p[0]) for p in walk(root / 'cpu', rel['cpu'], 'cpu.cfs_period_us')]
        cpus += [q / p for q, p in zip(quota, period) if q > 0 and p > 0]
    if 'memory' in rel:
        mem += [int(m[0]) for m in walk(root / 'memory', rel['memory'], 'memory.limit_in_bytes') if int(m[0]) < 2 ** 60]
    return min(cpus, default=None), min(mem, default=None)


def local_hardware(root=Path('/sys/fs/cgroup'), proc=Path('/proc/self/cgroup')) -> str:
    gpus = ''
    if shutil.which('nvidia-smi'):
        gpus = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    visible = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    quota, limit = cgroup_limits(root, proc)
    cpu = f'{min(visible, quota):g} CPU (cgroup quota; {visible} visible)' if quota and quota < visible else f'{visible} CPU'
    if limit is None:
        try:
            limit = int(next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal')).split()[1]) * 1024
            memory = f'{limit / 2 ** 30:.1f} GiB'
        except (OSError, StopIteration):
            memory = ''
    else:
        memory = f'{limit / 2 ** 30:.1f} GiB (cgroup limit)'
    return ', '.join(filter(None, [gpus.replace('\n', ', '), cpu, memory]))


def onprem_ledger(row):
    """A stage's Modal spend ledger as an on-prem in-process record: every *Usd field None, rate source dropped."""
    if isinstance(row, list):
        return [onprem_ledger(v) for v in row]
    if not isinstance(row, dict):
        return row
    return {k: None if 'usd' in k.lower() else 'on-prem in-process' if k == 'mode' else onprem_ledger(v)
            for k, v in row.items() if k != 'rateSource'}


def rewrite_ledgers():
    real = Path.write_text
    hardware = local_hardware()

    def write_text(self, data, *args, **kwargs):
        if self.name.endswith('spend-ledger.json'):
            try:
                row = json.loads(data)
            except ValueError:
                row = None
            if isinstance(row, dict):
                row = onprem_ledger(row)
                row.update(mode='on-prem in-process', hardware=hardware, modalHardwareProfile=row.get('modalHardwareProfile', row.get('hardware')),
                           note='run by scripts/onprem/run_stage.py in this process: no Modal call, no Modal charge; seconds are local')
                data = json.dumps(row, indent=2) + '\n'
        return real(self, data, *args, **kwargs)

    Path.write_text = write_text


# ---------------------------------------------------------------- the app, in-process
def image_mounts(app_path: Path, repo: Path | None) -> list:
    """(container path, repository path) of every add_local_file / add_local_dir(REPO / 'literal', '/container/path')."""
    out = []
    for node in ast.walk(ast.parse(app_path.read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('add_local_file', 'add_local_dir') and len(node.args) >= 2):
            continue
        src, dst = node.args[0], node.args[1]
        if (repo and isinstance(src, ast.BinOp) and isinstance(src.op, ast.Div) and isinstance(src.left, ast.Name)
                and src.left.id == 'REPO' and isinstance(src.right, ast.Constant) and isinstance(dst, ast.Constant)):
            out.append((dst.value, repo / src.right.value))
    return out  # ponytail: literal REPO / 'path' only; loops over directory names are left to the docker/ images


def map_mounts(pairs: list) -> Path | None:
    """Container paths that do not exist here become symlinks under a temporary root, reached through an import hook."""
    missing = [(dst, src) for dst, src in pairs if not Path(dst).exists() and src.exists()]
    if not missing:
        return None
    root = Path(tempfile.mkdtemp(prefix='onprem-mounts-'))
    for dst, src in missing:
        link = root / dst.lstrip('/'); link.parent.mkdir(parents=True, exist_ok=True); link.symlink_to(src.resolve())
    loaders = [(importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES),
               (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES),
               (importlib.machinery.SourcelessFileLoader, importlib.machinery.BYTECODE_SUFFIXES)]

    def hook(entry):
        mapped = root / str(entry).lstrip('/')
        if str(entry).startswith('/') and not Path(entry).exists() and mapped.is_dir():
            return importlib.machinery.FileFinder(str(mapped), *loaders)
        raise ImportError(entry)

    sys.path_hooks.insert(0, hook)
    sys.path_importer_cache.clear()
    return root


def load_app(path: Path):
    use_stub()
    path = path.resolve()
    for p in (str(Path.cwd()), str(path.parent.parent), str(path.parent)):  # `modal run` cwd + repo root + module dir
        if p not in sys.path:
            sys.path.insert(1, p)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def entrypoint_args(fn, argv):
    """`modal run` flags from the entrypoint signature: --snake-case, typed by annotation, required without a default."""
    parser = argparse.ArgumentParser(prog=fn.__name__)
    for name, p in inspect.signature(fn).parameters.items():
        kind = p.annotation if isinstance(p.annotation, type) else {'str': str, 'int': int, 'float': float, 'bool': bool}.get(str(p.annotation), str)
        flag = '--' + name.replace('_', '-')
        if kind is bool:
            parser.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=p.default if p.default is not p.empty else False)
        else:
            parser.add_argument(flag, dest=name, type=kind, required=p.default is p.empty, default=None if p.default is p.empty else p.default)
    return vars(parser.parse_args(argv))


def run(target: str, argv: list) -> object:
    use_stub()
    path, _, name = target.partition('::')
    path = Path(path)
    source = path.read_text()
    if 'local_entrypoint' not in source and 'modal.App' not in source:  # a driver script: run it as __main__
        sys.argv = [str(path), *argv]
        sys.path.insert(1, str(path.resolve().parent))
        return runpy.run_path(str(path), run_name='__main__')
    module = load_app(path)
    app = next(v for v in vars(module).values() if type(v).__name__ == 'App')
    map_mounts(image_mounts(path, Path(module.REPO) if hasattr(module, 'REPO') else None))
    entrypoints = app.registered_entrypoints
    if not entrypoints:
        raise SystemExit(f'{path} has no local entrypoint')
    raw = entrypoints[name] if name else next(iter(entrypoints.values()))
    return raw(**entrypoint_args(raw, argv))


# ---------------------------------------------------------------- the licence-clean geometry route, one stage
ROUTE_CERT = .05  # RoMa's own sample_thresh (certainty above it = certain); research-notes/geometry-backbone-ab-2026-10-06 mvs_route.py
FRAME_FILES = ('pts3d', 'conf', 'valid_mask', 'intrinsics', 'camera_to_world')


def geometry(argv) -> dict:
    """run_stage.py geometry --run RUN --start GEOM --output NEW (--weights DIR | --roma DIR) [--device cpu]

The licence-clean geometry route (research-notes/geometry-backbone-ab-2026-10-06), offline: the network is refused in this
process and HF_HUB_OFFLINE=1. GEOM = the start geometry (the DA3-BASE stage's output, RUN/geometry frame schema) of RUN's frames
(manifest.json frames[]: frame_id, canonical, alpha, content_rect_xyxy) ->
  RoMa v1 outdoor sparse + dense matches of every frame pair on the content rect (modal_apps/geometry_clean_ab.roma_pair):
    --weights DIR  run RoMa here from DIR = scripts/onprem/fetch_weights_geometry.py --cache DIR (the mirror layout); both files
                   are SHA-256 checked against the pins before loading and refused on a mismatch (roma_model); written to NEW/roma/
    --roma DIR     reuse RoMa outputs instead: roma-I-J.npz (uvA, uvB) and dense-I-J.npz (uvAB, certA, uvBA, certB), I < J in
                   frame order (the same files NEW/roma/ holds)
  -> bundle adjustment with one focal per photo (geometry_clean_ab.refine: the numpy Schur LM, modal_apps/bundle_adjust.py);
     a pass that is not usable (bundle_adjust doc) is refused and nothing is written
  -> two-view triangulation of the dense warp (geometry_clean_ab.mvs) at CERT = ROUTE_CERT, conf = 1 on every kept pixel
  -> NEW/frames/<frame_id>/{pts3d,conf,valid_mask,intrinsics,camera_to_world}.npy + canonical.png + candidate_manifest.json."""
    ap = argparse.ArgumentParser(prog='run_stage.py geometry', description=geometry.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--start', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--weights', type=Path)
    src.add_argument('--roma', type=Path)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args(argv)
    if a.output.exists():
        raise SystemExit(f'onprem geometry: {a.output} exists; choose a new directory')
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PANOPTES_ONPREM='1')
    block_network(); use_stub(); pin_torch_hub()
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'modal_apps')]
    import numpy as np
    import geometry_clean_ab as gc
    t0 = time.monotonic(); fman = json.loads((a.run / 'manifest.json').read_text())['frames']
    if sorted(p.name for p in (a.start / 'frames').iterdir()) != [f['frame_id'] for f in fman]:
        raise SystemExit(f'onprem geometry: {a.start}/frames are not the frames of {a.run}/manifest.json')
    frames = gc.read_geometry(a.start); H, W = frames[0]['pts3d'].shape[:2]
    rects = {tuple(f['content_rect_xyxy']) for f in fman}; x0, y0, x1, y1 = rects.pop()
    if rects or (y0, y1) != (0, H):
        raise SystemExit('onprem geometry: one content rect [x0, 0, x1, H] shared by every frame is required')
    contents = [fr['valid'] & np.load(a.run / f['alpha']) & np.isfinite(fr['pts3d']).all(-1) & (np.linalg.norm(fr['pts3d'], axis=-1) > 1e-6)
                & (fr['conf'] >= .1) for fr, f in zip(frames, fman)]  # prepare_capture_evidence.RULE (fair_ab.content_mask)
    pairs, keys, written = list(combinations(range(len(frames)), 2)), ('uvAB', 'certA', 'uvBA', 'certB'), {}
    if a.roma:
        matches = {(i, j): tuple(np.load(a.roma / f'roma-{i}-{j}.npz')[k] for k in ('uvA', 'uvB')) for i, j in pairs}
        dense = {(i, j): tuple(np.load(a.roma / f'dense-{i}-{j}.npz')[k] for k in keys) for i, j in pairs}
        used = [f'{kind}-{i}-{j}.npz' for i, j in pairs for kind in ('roma', 'dense')]
        roma = dict(source=f'reused {a.roma}', sha256={n: hashlib.sha256((a.roma / n).read_bytes()).hexdigest() for n in used})
    else:
        import torch
        from PIL import Image
        import fetch_weights_geometry as fwg
        torch.manual_seed(0); model = gc.roma_model(a.device, a.weights)  # SHA-256 checked first, a mismatch is refused
        crops = [Image.open(a.run / f['canonical']).convert('RGB').crop((x0, 0, x1, H)) for f in fman]
        matches, dense = {}, {}
        for i, j in pairs:
            r = gc.roma_pair(model, crops[i], crops[j], a.device, x0=x0)
            matches[(i, j)], dense[(i, j)] = (r['sparse']['uvA'], r['sparse']['uvB']), tuple(r['dense'][k] for k in keys)
            written[f'roma-{i}-{j}.npz'], written[f'dense-{i}-{j}.npz'] = gc.npz(**r['sparse']), gc.npz(**r['dense'])
        roma = dict(source=f'computed on {a.device} from {a.weights}', code=fwg.ROMA_CODE, torch=str(torch.__version__),
                    weightsSha256={rel: f[1] for n in ('roma_outdoor', 'roma_dinov2') for rel, f in fwg.FILES[n]['files'].items()})
    gc.MVS.update(CERT=ROUTE_CERT)
    baf, rep = gc.refine(frames, contents, matches, focal=True, size=(H, W))
    if not rep['usable']:
        raise SystemExit('onprem geometry: bundle adjustment not usable, nothing written: ' + json.dumps([p['report'] for p in rep['ba']]))
    mv, st = gc.mvs(baf, dense, x0=x0, w=x1 - x0, size=(H, W))
    for fr, f in zip(mv, fman):
        d = a.output / 'frames' / f['frame_id']; d.mkdir(parents=True)
        for name, v in zip(FRAME_FILES, (fr['pts3d'], fr['valid'].astype(np.float32), fr['valid'], fr['K'], fr['c2w'])):
            np.save(d / f'{name}.npy', v)
        shutil.copyfile(a.run / f['canonical'], d / 'canonical.png')
    for name, data in written.items():
        (a.output / 'roma').mkdir(exist_ok=True); (a.output / 'roma' / name).write_bytes(data)
    start = a.start / 'candidate_manifest.json'
    summary = dict(stage='run_stage.py geometry', run=str(a.run), start=str(a.start),
                   startModel=json.loads(start.read_text()).get('model_id') if start.exists() else None,
                   ba=rep['ba_backend'], baConverged=rep['converged'], baUsable=rep['usable'],
                   baPasses=[{k: p[k] for k in ('termination', 'iterations', 'cost', 'points', 'dropped')} for p in rep['ba']],
                   roma=roma, thresholds=dict(gc.MVS), conf='1 on kept pixels', contentRect=[x0, 0, x1, H], mvs=st,
                   seconds=round(time.monotonic() - t0, 1), created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (a.output / 'candidate_manifest.json').write_text(json.dumps(dict(summary, baReport=rep), indent=1, default=float) + '\n')
    print('onprem-geometry ' + json.dumps(summary, default=float))
    return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['geometry']:
        return geometry(argv[1:])
    opts = dict(offline=None, record=None, weights=None)
    while argv and argv[0].startswith('--') and argv[0][2:] in (*opts, 'self-test'):
        flag = argv.pop(0)[2:]
        if flag == 'self-test':
            return _check()
        opts[flag] = Path(argv.pop(0))
    if not argv:
        raise SystemExit(__doc__)
    os.environ['PANOPTES_ONPREM'] = '1'
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))  # docker stop: still remove the overlays (job copies)
    report = dict(target=argv[0], modalClient=use_stub().__file__, modalCredentials=any(k.startswith('MODAL_TOKEN') for k in os.environ))
    pin_torch_hub()
    if opts['weights']:
        os.environ['HF_HUB_OFFLINE'] = '1'
        report['weights'] = link_weights(opts['weights'].resolve())
    served = None
    if opts['offline']:
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        block_network()
        served = replay_urls(opts['offline'].resolve())
    elif opts['record']:
        record_urls(opts['record'].resolve())
    rewrite_ledgers()
    try:
        run(argv[0], argv[1:])
    finally:
        cleanup()
    report.update(offline=bool(opts['offline']), hfHubOffline=os.environ.get('HF_HUB_OFFLINE') == '1',
                  urlsServedFromBundle=len(served) if served is not None else None)
    print('onprem-run ' + json.dumps(report))


HUB_CHECK = r'''
import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import run_stage; run_stage.pin_torch_hub()
import torch.hub as hub
assert hub.load('acme/tools', 'm', 1, k=2) == (hub.get_dir() + '/acme_tools_main', 'local', 'm', (1,), {'k': 2})
assert hub.load('acme/tools:v1', 'm')[0].endswith('acme_tools_v1') and hub.load('/x/y', 'm', source='local')[:2] == ('/x/y', 'local')
assert hub.load_state_dict_from_url('file:///weights/w.pth') == 'copied file:///weights/w.pth'
for bad in (lambda: hub.load('other/repo', 'm'), lambda: hub.load_state_dict_from_url('https://dl.example/w.pth')):
    try:
        bad(); raise AssertionError('not refused')
    except RuntimeError as e:
        assert 'onprem' in str(e), e
print('hub ok')
'''
FAKE_TORCH_HUB = '''import os
def get_dir(): return os.environ['FAKE_HUB']
def load(repo_or_dir, model, *args, source='github', trust_repo=None, force_reload=False, verbose=True, skip_validation=False, **kwargs):
    if source == 'github': raise AssertionError('would probe GitHub')
    return (repo_or_dir, source, model, args, kwargs)
def download_url_to_file(url, dst, hash_prefix=None, progress=True): return 'copied ' + url
def load_state_dict_from_url(url, model_dir=None, **kw): return download_url_to_file(url, '/dev/null')
'''


def _check():
    """Synthetic apps under the stub: entrypoint, .remote/.map/.spawn, an @app.cls with @enter, a mounted helper module, a volume;
    record then replay one URL offline; weights overlay (job copies outside the weights, removed at the end); torch.hub pin;
    loopback parsing; cgroup CPU / memory limits."""
    import http.server
    import threading
    env_ok = not any(k.startswith('MODAL_TOKEN') for k in os.environ)
    real_write = Path.write_text
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'scripts').mkdir(); (tmp / 'apps').mkdir(); (tmp / 'vol').mkdir()
        (tmp / 'scripts/helper.py').write_text('def twice(x):\n    return 2 * x\n')
        (tmp / 'apps/demo.py').write_text(f'''
from pathlib import Path
import json, os, modal
assert modal.ONPREM_STUB
REPO = Path(__file__).resolve().parents[1]
app = modal.App('onprem-selftest')
image = modal.Image.debian_slim().pip_install('x').add_local_file(REPO / 'scripts/helper.py', '/onprem-selftest-mount/helper.py')
vol = modal.Volume.from_name('onprem-selftest-vol')
@app.function(image=image, volumes={{{json.dumps(str(tmp / 'vol'))}: vol}}, secrets=[modal.Secret.from_name('s')])
def work(x: int) -> int:
    import sys; sys.path.insert(0, '/onprem-selftest-mount'); import helper
    vol.reload(); Path({json.dumps(str(tmp / 'vol'))}, 'out.txt').write_text(str(helper.twice(x))); vol.commit()
    return helper.twice(x)
@app.cls(image=image, gpu='A100')
class Model:
    loads = 0
    @modal.enter()
    def load(self):
        type(self).loads += 1; self.k = 3
    @modal.method()
    def mul(self, x):
        return self.k * x
@app.local_entrypoint()
def main(x: int, label: str = 'a', loud: bool = False):
    assert os.environ['PANOPTES_ONPREM'] == '1'
    with app.run():
        assert work.remote(x) == 2 * x and list(work.map([1, 2])) == [2, 4] and work.spawn(3).get() == 6
        m = Model(); assert m.mul.remote(2) == 6 and list(m.mul.map([1, 2])) == [3, 6] and m.loads == 1
    print('demo', label, loud, b''.join(vol.read_file('out.txt')).decode())
    Path({json.dumps(str(tmp / 'vol'))}, 'spend-ledger.json').write_text(json.dumps({{'mode': 'ephemeral modal run', 'hardware': 'A100-80GB',
        'functionSeconds': 2.0, 'estimateUsd': 0.01, 'actualBilledUsd': None, 'rateSource': 'https://modal.com/pricing',
        'calls': [{{'mode': 'ephemeral modal run', 'callEstimateUsd': 0.02}}]}}))
''')
        out = io.StringIO()
        from contextlib import redirect_stdout
        try:
            with redirect_stdout(out):
                main([str(tmp / 'apps/demo.py'), '--x', '5', '--label', 'b', '--loud'])
        finally:
            Path.write_text = real_write
        assert out.getvalue().splitlines()[0] == 'demo b True 6', out.getvalue()
        assert sys.modules['modal'].ONPREM_STUB and str(STUB) in os.environ['PYTHONPATH']
        ledger = json.loads((tmp / 'vol/spend-ledger.json').read_text())
        assert ledger['mode'] == ledger['calls'][0]['mode'] == 'on-prem in-process' and ledger['modalHardwareProfile'] == 'A100-80GB', ledger
        assert ledger['estimateUsd'] is None and ledger['calls'][0]['callEstimateUsd'] is None and 'rateSource' not in ledger, ledger
        assert ledger['functionSeconds'] == 2.0 and 'CPU' in ledger['hardware'], ledger
        # record from a loopback HTTP server, then replay with it stopped and the network blocked
        (tmp / 'www').mkdir(); (tmp / 'www/layer.json').write_text('{"ok": 1}')
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(tmp / 'www'), **k)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{server.server_address[1]}/layer.json'
        real = urllib.request.urlopen
        record_urls(tmp / 'bundle')
        assert json.loads(urllib.request.urlopen(url, timeout=5).read()) == {'ok': 1}
        server.shutdown(); server.server_close()
        urllib.request.urlopen = real
        saved = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
        replay_urls(tmp / 'bundle'); block_network()
        try:
            assert json.loads(urllib.request.urlopen(url).read()) == {'ok': 1}
            for bad in (lambda: urllib.request.urlopen('https://example.com/x'),
                        lambda: socket.create_connection(('93.184.216.34', 80), timeout=2),
                        lambda: socket.getaddrinfo('localhost.example.com', 80)):
                try:
                    bad(); raise AssertionError('network was not refused')
                except (OSError, urllib.error.URLError) as error:
                    assert 'onprem offline' in str(error), error
        finally:
            urllib.request.urlopen = real; socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = saved
        for host, ok in (('localhost', True), ('', True), (None, True), ('127.0.0.1', True), ('127.8.9.1', True), ('::1', True),
                         ('[::1]', True), ('::ffff:127.0.0.1', True), ('0.0.0.0', True), (b'127.0.0.1', True),
                         ('localhost.evil.example', False), ('127.example.com', False), ('::1.evil', False), ('10.0.0.1', False),
                         ('huggingface.co', False), ('0.0.0.0.example', False)):
            assert _loopback(host) == ok, (host, ok)
        # weights overlay: the stage's job copy lands in scratch, never in the weights; all of it is gone after cleanup()
        cache, link = tmp / 'weights', tmp / 'container/cache'
        (cache / 'recgen/weights').mkdir(parents=True); (cache / 'recgen/weights/w.bin').write_bytes(b'w')
        (cache / 'recgen/stage.json').write_text('{}'); (cache / 'recgen/jobs-from-an-old-run').mkdir()
        (cache / 'manifest.json').write_text(json.dumps({'hfHome': 'hf', 'models': {'recgen': {
            'path': 'recgen', 'link': str(link), 'snapshot': 'recgen/weights', 'files': {'w.bin': {}, 'stage.json': {}},
            'extras': {'stage.json': 'recgen/stage.json'}}, 'plain': {'path': 'hf/hub', 'snapshot': 'hf/hub/x', 'files': {}}}}))
        link.parent.mkdir(); link.symlink_to(cache / 'recgen')  # an older direct link is replaced
        done = link_weights(cache)
        assert list(done) == ['recgen'] and sorted(p.name for p in link.iterdir()) == ['stage.json', 'weights'], done
        assert (link / 'weights/w.bin').read_bytes() == b'w'
        (link / 'jobs/abc').mkdir(parents=True); (link / 'jobs/abc/input.npz').write_bytes(b'photo')
        overlay = Path(os.readlink(link))
        assert (overlay / 'jobs/abc/input.npz').is_file() and not (cache / 'recgen/jobs').exists()
        try:
            link_weights(cache); raise AssertionError('a live overlay was taken over')
        except RuntimeError as error:
            assert 'another run_stage' in str(error), error
        cleanup()
        assert not link.exists() and not link.is_symlink() and not overlay.parent.exists()
        assert (cache / 'recgen/weights/w.bin').read_bytes() == b'w' and not (cache / 'recgen/jobs').exists()
        link.symlink_to(tmp / 'missing'); link_weights(cache); cleanup()  # a dangling link (crash) is replaced
        # torch.hub pin, in a child process with a fake torch package
        (tmp / 'fake/torch').mkdir(parents=True); (tmp / 'fake/torch/__init__.py').write_text('from . import hub\n')
        (tmp / 'fake/torch/hub.py').write_text(FAKE_TORCH_HUB); (tmp / 'hub/acme_tools_main').mkdir(parents=True); (tmp / 'hub/acme_tools_v1').mkdir()
        p = subprocess.run([sys.executable, '-c', HUB_CHECK, str(Path(__file__).resolve().parent), str(tmp / 'fake')],
                           env={**os.environ, 'FAKE_HUB': str(tmp / 'hub')}, capture_output=True, text=True)
        assert p.returncode == 0 and 'hub ok' in p.stdout, p.stderr
        # cgroup limits: v2 quota 2.5 CPUs and 4 GiB on a parent cgroup, v1 quota, unlimited
        cg = tmp / 'cg'; (cg / 'a/b').mkdir(parents=True); (tmp / 'p2').write_text('0::/a/b\n')
        (cg / 'a/cpu.max').write_text('250000 100000\n'); (cg / 'a/b/cpu.max').write_text('max 100000\n'); (cg / 'a/b/memory.max').write_text(f'{4 * 2 ** 30}\n')
        assert cgroup_limits(cg, tmp / 'p2') == (2.5, 4 * 2 ** 30)
        (cg / 'cpu/docker/x').mkdir(parents=True); (tmp / 'p1').write_text('5:cpu,cpuacct:/docker/x\n4:memory:/docker/x\n')
        (cg / 'cpu/docker/x/cpu.cfs_quota_us').write_text('400000'); (cg / 'cpu/docker/x/cpu.cfs_period_us').write_text('100000')
        (cg / 'memory/docker/x').mkdir(parents=True); (cg / 'memory/docker/x/memory.limit_in_bytes').write_text(str(2 ** 63 - 4096))
        assert cgroup_limits(cg, tmp / 'p1') == (4.0, None) and cgroup_limits(cg, tmp / 'none') == (None, None)
        assert 'CPU' in local_hardware(cg, tmp / 'p2')
    print('run_stage self-test passed: modal stub (entrypoint, .remote/.map/.spawn, @app.cls + @enter, image mount, volume),',
          'record/replay offline, loopback by parsed address, weights overlay (job copy outside the weights, removed),',
          'torch.hub vendored + network downloads refused (file:// allowed), cgroup CPU/memory limits, spend ledger -> on-prem',
          'in-process without USD;',
          'modal credentials in env:', not env_ok)


if __name__ == '__main__':
    main()
