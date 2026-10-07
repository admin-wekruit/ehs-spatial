"""serving/registry.yaml: every model service (file, port, image, GPU stack, weights + pins, licence, VRAM). Read by the
services (/v1/info, the start-up weight check) and by the Makefile:
  python serving/registry.py services a         -> the services of GPU stack a, one per line (make up GPU=a)
  python serving/registry.py info sam3d         -> the /v1/info document (code_sha from git)
  python serving/registry.py check              -> self-test
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

PATH = Path(__file__).with_name('registry.yaml')
INFO_KEYS = ('model_id', 'model_revision', 'code_revision', 'weights_sha256', 'licence', 'code_sha', 'seed_policy')


def load(path: Path = PATH) -> dict:
    models = yaml.safe_load(path.read_text())['models']
    for name, m in models.items():
        m['name'] = name
    return models


def entry(model: str) -> dict:
    models = load()
    if model not in models:
        raise KeyError(f'{model} is not in {PATH} ({sorted(models)})')
    return models[model]


def weights_sha256(e: dict) -> dict:
    """{file name: sha256} over the model's weight sets; a name must be unique across the sets."""
    out = {}
    for ws in e.get('weights', []):
        for rel, sha in (ws.get('sha256') or {}).items():
            name = Path(rel).name
            if name in out and out[name] != sha:
                raise ValueError(f'{e["name"]}: weight file name {name} is not unique across its sets')
            out[name] = sha
    return out


def info(e: dict, code_sha: str = 'unknown') -> dict:
    return dict(model_id=e['model_id'], model_revision=e.get('model_revision'), code_revision=e.get('code_revision'),
                weights_sha256=weights_sha256(e), licence=e['licence'], code_sha=code_sha, seed_policy=e.get('seed_policy'),
                # every weight set behind the service by name (geometry-mvs: DA3-BASE, RoMa, DINOv2, MoGe-3; sam3d: + DINOv2)
                components={ws['name']: dict(repo=ws['repo'], revision=ws.get('revision')) for ws in e.get('weights', [])})


def services(gpu: str) -> list[str]:
    return [name for name, m in load().items() if str(m.get('gpu')) == gpu]


def _check():
    models = load()
    assert set(models) == {'sam3', 'mapanything', 'moge', 'geometry-mvs', 'sam3d'}, sorted(models)
    assert sorted(services('a')) == ['sam3', 'sam3d'] and sorted(services('b')) == ['geometry-mvs', 'mapanything', 'moge']
    assert len({m['port'] for m in models.values()}) == 5
    for m in models.values():
        for key in ('service', 'port', 'image', 'gpu', 'model_id', 'licence', 'vram_gb', 'weights'):
            assert key in m, (m['name'], key)
        assert (Path(__file__).resolve().parents[1] / m['service']).is_file(), m['service']
        doc = info(m, 'abc')
        assert tuple(doc)[:7] == INFO_KEYS, tuple(doc)
    assert set(info(models['sam3d'])['weights_sha256']) == {'ss_generator.ckpt', 'slat_generator.ckpt', 'ss_decoder.ckpt', 'slat_decoder_mesh.ckpt', 'dinov2_vitl14_reg4_pretrain.pth'}
    assert set(info(models['geometry-mvs'])['components']) == {'da3-base', 'roma_outdoor', 'roma_dinov2', 'moge3'}
    print('registry self-test passed')


if __name__ == '__main__':
    cmd = sys.argv[1:2]
    if cmd == ['services']:
        print('\n'.join(services(sys.argv[2])))
    elif cmd == ['info']:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import v1_common
        print(json.dumps(info(entry(sys.argv[2]), v1_common.code_sha()), indent=1))
    elif cmd == ['check']:
        _check()
    else:
        raise SystemExit(__doc__)
