"""Run any Modal app (or driver script) of the workcell photo pipeline in this process, without Modal: the on-prem runner.

  python scripts/onprem/run_stage.py [--offline BUNDLE | --record BUNDLE] [--weights DIR] APP.py[::ENTRYPOINT] [flags as for `modal run`]
  python scripts/onprem/run_stage.py [...] SCRIPT.py [its own arguments]          (a file without a Modal local entrypoint)
  python scripts/onprem/run_stage.py --self-test

The module is imported as `modal run` imports it. The `modal` pip package is only a library here: no account, token, config or
network is used (verified with every MODAL_* variable unset and MODAL_CONFIG_PATH pointing at a missing file). Then:
- the app's local entrypoint runs unchanged, and every Function.remote / spawn / map / starmap runs the function body here
  (modal's own Function.local: the raw function, class methods after their @enter);
- an image's add_local_file / add_local_dir container paths (`REPO / 'x'` -> '/check/y') resolve to the repository files
  through an import hook; images built from docker/ already hold those paths, then nothing is mapped;
- a Volume is the local directory at its container mount path: commit / reload are no-ops, read_file and batch_upload use it;
- --record BUNDLE saves every urllib response the stage reads (index.json + blobs/<sha256>); --offline BUNDLE replays them,
  refuses every other URL and every non-loopback socket / DNS lookup in this process, and sets HF_HUB_OFFLINE=1;
- --weights DIR (written by fetch_weights.py) links each model to the path its stage expects and sets HF_HOME.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
import sys
import tempfile
import urllib.request
import urllib.response

LOOPBACK = ('localhost', '127.', '::1', '0.0.0.0')


# ---------------------------------------------------------------- network: record / offline replay
def _loopback(host) -> bool:
    host = str(host or '')
    return host.startswith(LOOPBACK) or host == ''


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


# ---------------------------------------------------------------- weights cache (fetch_weights.py manifest)
def link_weights(cache: Path) -> dict:
    manifest = json.loads((cache / 'manifest.json').read_text())
    os.environ.setdefault('HF_HOME', str(cache / manifest.get('hfHome', 'hf')))
    os.environ.setdefault('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))
    done = {}
    for name, model in manifest['models'].items():
        link = model.get('link')
        if not link:
            continue
        target, path = cache / model['path'], Path(link)
        if path.is_symlink() and Path(os.readlink(path)) == target:
            pass
        elif path.exists():  # a real directory a deployment mounted there itself: leave it
            print(f'onprem: {link} exists, not linking {name}', file=sys.stderr)
            continue
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target, target_is_directory=True)
        done[name] = f'{link} -> {target}'
    return done


# ---------------------------------------------------------------- modal: in-process execution
class _Done:
    def __init__(self, value):
        self.value = value

    def get(self, timeout=None):
        return self.value

    def __iter__(self):
        return iter(self.value)


def patch_modal(volume_paths: dict):
    """Function calls run here; volumes are the local directories at their container mount paths."""
    import warnings
    import modal
    warnings.filterwarnings('ignore', message='.*executing locally and will not have access to the mounted Volume')

    def remote(self, *args, **kwargs):
        return self.local(*args, **kwargs)

    modal.Function.remote = remote
    modal.Function.spawn = lambda self, *a, **k: _Done(self.local(*a, **k))
    modal.Function.map = lambda self, *iters, **k: (self.local(*args) for args in zip(*iters))
    modal.Function.starmap = lambda self, items, **k: (self.local(*args) for args in items)
    modal.Volume.commit = modal.Volume.reload = lambda self, *a, **k: None
    modal.App.run = lambda self, *a, **k: contextlib.nullcontext(self)  # driver scripts: `with app.run():` needs no server

    def mount(volume):
        name = getattr(volume, 'name', None)
        if name not in volume_paths:  # a driver script imported the app after start-up: read its functions' mounts now
            volume_paths.update(app_volumes(*[v for m in list(sys.modules.values()) for v in vars(m).values() if type(v).__name__ == 'App']))
        if name not in volume_paths:
            raise RuntimeError(f'onprem: volume {name} is not mounted by any function of this app')
        return Path(volume_paths[name])

    def read_file(self, path, *a, **k):
        yield (mount(self) / str(path).lstrip('/')).read_bytes()

    class Upload:
        def __init__(self, root):
            self.root = root

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def put_file(self, local, remote, *a, **k):
            dest = self.root / str(remote).lstrip('/'); dest.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(local, 'read'):
                dest.write_bytes(local.read())
            else:
                shutil.copyfile(local, dest)

        def put_directory(self, local, remote, *a, **k):
            shutil.copytree(local, self.root / str(remote).lstrip('/'), dirs_exist_ok=True)

    modal.Volume.read_file = read_file
    modal.Volume.batch_upload = lambda self, *a, **k: Upload(mount(self))


def app_volumes(*apps) -> dict:
    """Volume name -> container mount path, from every function of the given apps."""
    return {getattr(v, '_name', None): str(path) for app in apps for fn in app.registered_functions.values()
            for path, v in fn.spec.volumes.items()}


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
    path = path.resolve()
    for p in (str(Path.cwd()), str(path.parent.parent), str(path.parent)):  # `modal run` cwd + repo root + module dir
        if p not in sys.path:
            sys.path.insert(0, p)
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
    path, _, name = target.partition('::')
    path = Path(path)
    source = path.read_text()
    if 'local_entrypoint' not in source and 'modal.App' not in source:  # a driver script: run it as __main__ with the patches
        patch_modal({})
        sys.argv = [str(path), *argv]
        sys.path.insert(0, str(path.resolve().parent))
        return runpy.run_path(str(path), run_name='__main__')
    module = load_app(path)
    app = next(v for v in vars(module).values() if type(v).__name__ == 'App')
    patch_modal(app_volumes(app))
    map_mounts(image_mounts(path, Path(module.REPO) if hasattr(module, 'REPO') else None))
    entrypoints = app.registered_entrypoints
    if not entrypoints:
        raise SystemExit(f'{path} has no local entrypoint')
    entry = entrypoints[name] if name else next(iter(entrypoints.values()))
    raw = entry.info.raw_f
    return raw(**entrypoint_args(raw, argv))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = dict(offline=None, record=None, weights=None)
    while argv and argv[0].startswith('--') and argv[0][2:] in (*opts, 'self-test'):
        flag = argv.pop(0)[2:]
        if flag == 'self-test':
            return _check()
        opts[flag] = Path(argv.pop(0))
    if not argv:
        raise SystemExit(__doc__)
    report = dict(target=argv[0], modalCredentials=any(k.startswith('MODAL_TOKEN') for k in os.environ))
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
    run(argv[0], argv[1:])
    report.update(offline=bool(opts['offline']), hfHubOffline=os.environ.get('HF_HUB_OFFLINE') == '1',
                  urlsServedFromBundle=len(served) if served is not None else None)
    print('onprem-run ' + json.dumps(report))


def _check():
    """Synthetic app: a mounted helper module, a volume, an entrypoint calling .remote; record then replay one URL offline."""
    import http.server
    import threading
    env_ok = not any(k.startswith('MODAL_TOKEN') for k in os.environ)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'scripts').mkdir(); (tmp / 'apps').mkdir(); (tmp / 'vol').mkdir()
        (tmp / 'scripts/helper.py').write_text('def twice(x):\n    return 2 * x\n')
        (tmp / 'apps/demo.py').write_text(f'''
from pathlib import Path
import json, modal
REPO = Path(__file__).resolve().parents[1]
app = modal.App('onprem-selftest')
image = modal.Image.debian_slim().add_local_file(REPO / 'scripts/helper.py', '/onprem-selftest-mount/helper.py')
vol = modal.Volume.from_name('onprem-selftest-vol')
@app.function(image=image, volumes={{{json.dumps(str(tmp / 'vol'))}: vol}})
def work(x: int) -> int:
    import sys; sys.path.insert(0, '/onprem-selftest-mount'); import helper
    vol.reload(); Path({json.dumps(str(tmp / 'vol'))}, 'out.txt').write_text(str(helper.twice(x))); vol.commit()
    return helper.twice(x)
@app.local_entrypoint()
def main(x: int, label: str = 'a', loud: bool = False):
    with app.run():
        assert work.remote(x) == 2 * x and list(work.map([1, 2])) == [2, 4] and work.spawn(3).get() == 6
    print('demo', label, loud, b''.join(vol.read_file('out.txt')).decode())
''')
        out = io.StringIO()
        from contextlib import redirect_stdout
        with redirect_stdout(out):
            run(str(tmp / 'apps/demo.py'), ['--x', '5', '--label', 'b', '--loud'])
        assert out.getvalue().strip() == 'demo b True 6', out.getvalue()
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
                        lambda: socket.create_connection(('93.184.216.34', 80), timeout=2)):
                try:
                    bad(); raise AssertionError('network was not refused')
                except (OSError, urllib.error.URLError) as error:
                    assert 'onprem offline' in str(error), error
        finally:
            urllib.request.urlopen = real; socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = saved
    print('run_stage self-test passed: entrypoint + .remote/.map/.spawn in-process, image mount, volume, record/replay offline;',
          'modal credentials in env:', not env_ok)


if __name__ == '__main__':
    main()
