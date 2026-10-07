"""Fetch the pinned Depth Anything 3 checkpoints of docker/da3.Dockerfile once into a local cache, SHA-256 checked, with a manifest.

  python scripts/onprem/fetch_weights_da3.py --cache DIR [--models da3-large-1.1,da3-base]
  python scripts/onprem/fetch_weights_da3.py --cache DIR --verify          re-hash every file of DIR/manifest.json (offline)
  python scripts/onprem/fetch_weights_da3.py --self-test

The mechanics are fetch_weights.py's, imported unchanged (fetch / write_manifest / verify, layout 'local'): each checkpoint is
hf_hub_download-ed into DIR/local/<name> (model.safetensors + config.json), the folder serving scripts/candidate_geometry_backend.py
reads as --model-dir. DIR can be the same cache as fetch_weights.py (one manifest.json, merged). The only step that needs the
internet; the image sets HF_HUB_OFFLINE=1 and this script switches it off for itself (fetch_weights.fetch).
Pins = serving candidate_geometry_backend.py CHECKPOINTS (revision, model.safetensors SHA-256 = Hugging Face LFS oid) + config.json.

LICENCE - read before commercial use (checked 2026-10-05):
  da3-base       Apache-2.0: Hugging Face card and the official README model table agree.
  da3-large-1.1  DISPUTED: the Hugging Face card says apache-2.0, but the official README (ByteDance-Seed/Depth-Anything-3, every
                 revision including the pinned code 3d835ec) lists DA3-LARGE-1.1 as CC BY-NC 4.0, as is the predecessor DA3-LARGE's card.
                 Treat it as non-commercial until the authors confirm the licence in writing.
  code           Apache-2.0 (Depth-Anything-3 @ 3d835ec).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_weights as fw  # noqa: E402

CODE_REVISION = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'  # = candidate_geometry_backend.DA3_CODE_REVISION (docker/da3.Dockerfile)
MODELS = {
    'da3-large-1.1': dict(repo='depth-anything/DA3-LARGE-1.1', revision='0e109ae307c5982f319a67cf6f9f99ccdc0ec97c', layout='local',
                          path='local/da3-large-1.1', allow=['model.safetensors', 'config.json'],
                          sha256={'model.safetensors': '739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64',
                                  'config.json': '744dcaf53859490ed92fc6cb98d68d3daf624b8c54533aaf604bdb53f06321f5'},
                          licence='DISPUTED: HF card apache-2.0, official README CC BY-NC 4.0; treat as non-commercial until confirmed. Code Apache-2.0'),
    'da3-base': dict(repo='depth-anything/DA3-BASE', revision='f4a6c9b3c95e41c82048423d3493a81ec3fa810e', layout='local',
                     path='local/da3-base', allow=['model.safetensors', 'config.json'],
                     sha256={'model.safetensors': 'e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5',
                             'config.json': '5e34115ebc17bd2d8d43033c5f72e9446ac8833fd61d3fa160b7e67e0bb5b7b5'},
                     licence='Apache-2.0 (HF card and official README); code Apache-2.0'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--models', default='da3-large-1.1,da3-base')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return _check()
    cache = args.cache.resolve(); cache.mkdir(parents=True, exist_ok=True)
    if args.verify:
        print(json.dumps(fw.verify(cache)))
        return
    got = {}
    for name in args.models.split(','):
        got[name] = fw.fetch(cache, name, 'hub', MODELS[name])
        print(json.dumps({name: {k: got[name][k] for k in ('revision', 'bytes', 'seconds', 'licence')}}), flush=True)
    fw.write_manifest(cache, got)
    print(json.dumps({'manifest': str(cache / 'manifest.json'), 'models': sorted(got)}))


def _check():
    """Pins equal the runner's CHECKPOINTS (when serving is importable); a stubbed hub download goes through fetch_weights' 'local'
    path into a manifest that --verify accepts, and a wrong pin or a changed file is refused."""
    import hashlib
    import shutil
    import tempfile
    serving = [p for p in (os.environ.get('PANOPTES_SERVING', ''), '/serving') if p and Path(p, 'scripts/candidate_geometry_backend.py').exists()]
    if serving:
        sys.path.insert(0, str(Path(serving[0], 'scripts')))
        import candidate_geometry_backend as cgb
        assert cgb.DA3_CODE_REVISION == CODE_REVISION, cgb.DA3_CODE_REVISION
        for name, spec in MODELS.items():
            _, rev, sha = cgb.CHECKPOINTS[spec['repo']]
            assert (rev, sha) == (spec['revision'], spec['sha256']['model.safetensors']), (name, rev, sha)
    import huggingface_hub
    real = huggingface_hub.hf_hub_download
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); src = tmp / 'src'; src.mkdir()
        blobs = {'model.safetensors': b'weights' * 999, 'config.json': b'{"model_name": "da3-large"}'}
        for f, data in blobs.items():
            (src / f).write_bytes(data)
        spec = dict(MODELS['da3-large-1.1'], sha256={f: hashlib.sha256(d).hexdigest() for f, d in blobs.items()})

        def fake(repo, filename, revision=None, local_dir=None, **kw):
            assert (repo, revision) == (spec['repo'], spec['revision'])
            Path(local_dir).mkdir(parents=True, exist_ok=True); return shutil.copy(src / filename, Path(local_dir) / filename)
        huggingface_hub.hf_hub_download = fake
        try:
            os.environ['HF_HUB_OFFLINE'] = '1'  # as in the image; fetch must switch it off itself
            cache = tmp / 'cache'
            fw.write_manifest(cache, {'da3-large-1.1': fw.fetch(cache, 'da3-large-1.1', 'hub', spec)})
            assert os.environ['HF_HUB_OFFLINE'] == '0'
            m = json.loads((cache / 'manifest.json').read_text())['models']['da3-large-1.1']
            assert m['snapshot'] == 'local/da3-large-1.1' and set(m['files']) == set(blobs) and 'DISPUTED' in m['licence'], m
            assert fw.verify(cache)['da3-large-1.1']['files'] == 2
            try:
                fw.fetch(tmp / 'c2', 'da3-large-1.1', 'hub', dict(spec, sha256=dict(spec['sha256'], **{'config.json': '0' * 64})))
                raise AssertionError('wrong pin accepted')
            except ValueError:
                pass
            (cache / 'local/da3-large-1.1/config.json').write_bytes(b'{}')
            try:
                fw.verify(cache); raise AssertionError('changed file accepted')
            except ValueError:
                pass
        finally:
            huggingface_hub.hf_hub_download = real
    print('fetch_weights_da3 self-test passed: pins = runner CHECKPOINTS' + (' (checked)' if serving else ' (serving not found, skipped)')
          + '; local layout fetch -> manifest -> verify; wrong pin and changed file refused')


if __name__ == '__main__':
    main()
