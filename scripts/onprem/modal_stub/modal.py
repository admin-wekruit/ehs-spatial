"""On-prem stand-in for the `modal` client: the API the pipeline's Modal apps touch, executed in this process.

scripts/onprem/run_stage.py puts this directory first on sys.path, so the on-prem images need no modal client (and none of its
unpinned dependencies). Nothing here talks to a network, an account or a config file:
- App.function / App.cls / App.local_entrypoint register the user code; Function.remote / local / spawn / map / starmap call
  it here (class methods after the class's @enter methods, once per instance), App.run() is a no-op context;
- Image and Secret accept every builder call and build nothing (the image is docker/*.Dockerfile);
- a Volume is the local directory at the container path a function mounts it on (commit / reload are no-ops; read_file,
  batch_upload and remove_file act on that directory);
- deployed objects (Function.from_name, Cls.from_name) are refused: there is no Modal deployment on-prem.
Anything else is an AttributeError at import time, so an app that needs more of the API fails loudly instead of half-running.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
import shutil

__version__ = '0+onprem.stub'
ONPREM_STUB = True


def is_local() -> bool:
    return True


def _refuse(what):
    raise RuntimeError(f'onprem: {what} needs a Modal deployment; on-prem stages run in-process (scripts/onprem/run_stage.py)')


class _Builder:
    """Image / Secret / Retries ...: every builder call returns the same object; `with image.imports():` works too."""
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return lambda *a, **k: self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __repr__(self):
        return f'<onprem stub {self._name}>'


Image, Secret = _Builder('Image'), _Builder('Secret')


class _Done:
    def __init__(self, value):
        self.value = value

    def get(self, timeout=None):
        return self.value


class Function:
    def __init__(self, f):
        self.raw_f, self.__name__, self.__doc__ = f, getattr(f, '__name__', 'function'), getattr(f, '__doc__', None)

    def local(self, *args, **kwargs):
        return self.raw_f(*args, **kwargs)

    remote = __call__ = local

    def spawn(self, *args, **kwargs):
        return _Done(self.local(*args, **kwargs))

    def map(self, *iterables, kwargs=None, **_options):
        return (self.local(*args, **(kwargs or {})) for args in zip(*iterables))

    def starmap(self, items, kwargs=None, **_options):
        return (self.local(*args, **(kwargs or {})) for args in items)

    @classmethod
    def from_name(cls, *a, **k):
        _refuse('modal.Function.from_name')


class _Obj:
    """An instance of an @app.cls class: the user object is made, and its @enter methods run, on the first method call."""
    def __init__(self, user_cls, args, kwargs):
        self._user_cls, self._args, self._kwargs, self._instance = user_cls, args, kwargs, None

    def _user(self):
        if self._instance is None:
            self._instance = self._user_cls(*self._args, **self._kwargs)
            hooks = [f for c in reversed(self._user_cls.__mro__) for f in vars(c).values() if hasattr(f, '_onprem_enter')]
            for f in sorted(hooks, key=lambda f: not f._onprem_enter.get('snap')):  # snap=True hooks first, as modal runs them
                f(self._instance)
        return self._instance

    def __getattr__(self, attr):
        if attr.startswith('_'):
            raise AttributeError(attr)
        if not hasattr(getattr(self._user_cls, attr, None), '_onprem_method'):
            return getattr(self._user(), attr)
        return Function(lambda *a, **k: getattr(self._user(), attr)(*a, **k))


class Cls:
    def __init__(self, user_cls):
        self._user_cls = user_cls
        self.__name__ = user_cls.__name__

    def __call__(self, *args, **kwargs):
        return _Obj(self._user_cls, args, kwargs)

    @classmethod
    def from_name(cls, *a, **k):
        _refuse('modal.Cls.from_name')


def _marker(name):
    def decorator_factory(*args, **options):
        def mark(f):
            setattr(f, name, options)
            return f
        return mark(args[0]) if args and callable(args[0]) else mark
    return decorator_factory


enter, method = _marker('_onprem_enter'), _marker('_onprem_method')


class Volume:
    _by_name: dict = {}

    def __init__(self, name):
        self.name = self._name = name
        self.mount = None  # the container path of the first function that mounts it

    @classmethod
    def from_name(cls, name, *a, **k):
        return cls._by_name.setdefault(name, cls(name))

    def commit(self, *a, **k):
        pass

    reload = commit

    def _root(self) -> Path:
        if self.mount is None:
            raise RuntimeError(f'onprem: volume {self.name} is not mounted by any function of the imported apps')
        return Path(self.mount)

    def read_file(self, path, *a, **k):
        yield (self._root() / str(path).lstrip('/')).read_bytes()

    def remove_file(self, path, recursive=False, *a, **k):
        p = self._root() / str(path).lstrip('/')
        shutil.rmtree(p) if recursive and p.is_dir() else p.unlink()

    def batch_upload(self, *a, **k):
        return _Upload(self._root())


class _Upload:
    def __init__(self, root: Path):
        self.root = root

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def put_file(self, local, remote, *a, **k):
        dest = self.root / str(remote).lstrip('/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(local.read()) if hasattr(local, 'read') else shutil.copyfile(local, dest)

    def put_directory(self, local, remote, *a, **k):
        shutil.copytree(local, self.root / str(remote).lstrip('/'), dirs_exist_ok=True)


def _mount(volumes):
    for path, volume in (volumes or {}).items():
        if isinstance(volume, Volume) and volume.mount is None:
            volume.mount = str(path)


class App:
    def __init__(self, name=None, *a, **k):
        self.name = name
        self.registered_functions, self.registered_classes, self.registered_entrypoints = {}, {}, {}

    def function(self, *args, **options):
        def register(f):
            _mount(options.get('volumes'))
            fn = self.registered_functions[f.__name__] = Function(f)
            return fn
        return register(args[0]) if args and callable(args[0]) else register

    def cls(self, *args, **options):
        def register(user_cls):
            _mount(options.get('volumes'))
            c = self.registered_classes[user_cls.__name__] = Cls(user_cls)
            for name, f in vars(user_cls).items():
                if hasattr(f, '_onprem_method'):
                    self.registered_functions[f'{user_cls.__name__}.{name}'] = c
            return c
        return register(args[0]) if args and callable(args[0]) else register

    def local_entrypoint(self, *args, **options):
        def register(f):
            self.registered_entrypoints[f.__name__] = f
            return f
        return register(args[0]) if args and callable(args[0]) else register

    def run(self, *a, **k):
        return contextlib.nullcontext(self)
