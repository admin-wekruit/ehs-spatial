"""Fetch the weights of the licence-clean geometry route once into the same local cache as fetch_weights.py (one shared
manifest.json), SHA-256 checked and refused on mismatch; offline afterwards. The image never holds weights.

  python scripts/onprem/fetch_weights_geometry.py --cache DIR [--models roma_outdoor,roma_dinov2,da3-base,moge3]
  python scripts/onprem/fetch_weights_geometry.py --cache DIR --verify          re-hash every file of DIR/manifest.json (offline)
  python scripts/onprem/fetch_weights_geometry.py --self-test

Route (research-notes/geometry-backbone-ab-2026-10-06): DA3-BASE start (MoGe-3 the alternate) -> RoMa v1 outdoor dense matches
-> bundle adjustment with focal -> two-view triangulation of RoMa's dense warp. What lands where:
  roma_outdoor  DIR/roma/outdoor/roma_outdoor.pth + LICENSE         the exact URL romatch@77f8d68 roma_outdoor() downloads
  roma_dinov2   DIR/roma/dinov2/dinov2_vitl14_pretrain.pth + LICENSE the exact URL it downloads for its DINOv2 ViT-L/14 encoder
  da3-base      DIR/local/da3-base: config.json, model.safetensors, README.md (the HF card) = fetch_weights_da3.py's pins
  moge3         DIR/hf/hub snapshot (HF_HOME=DIR/hf)                 = fetch_weights.py's pin
romatch would otherwise download both RoMa files through torch.hub with no hash check. Pass them explicitly instead, each read
once and SHA-256 checked against FILES before torch.load (modal_apps/geometry_clean_ab.roma_model does this):
  roma_outdoor(device, use_custom_corr=False, **roma_state_dicts(DIR, device))   # = weights=..., dinov2_weights=...
Mirror: the Modal volume 'panoptes-geometry-weights' holds exactly this layout, manifest.json included (filled 2026-10-06 by
this script from the primary sources). Copy it to DIR with `modal volume get panoptes-geometry-weights / DIR`, then run --verify.
Runtime packages: docker/geometry-requirements.txt. It has no pycolmap: the PyPI 4.2.1 wheel statically links GPL-2.0+ SuiteSparse.

LICENCE - read before commercial use (checked 2026-10-06; not legal advice):
  roma_outdoor  MIT, inferred. No licence travels with the release asset. The Parskatt/storage repository that hosts it is MIT
                (its LICENSE, first line "MIT License", is fetched alongside). RoMa's README covers code only ("All our code
                except DINOv2 is MIT license."). Trained on MegaDepth; the training-data provenance is unaudited.
  roma_dinov2   Apache-2.0: DINOv2 README "code and model weights" and MODEL_CARD; its LICENSE is fetched alongside. The same
                dl.fbaipublicfiles.com/dinov2 tree also hosts FAIR non-commercial models: mirror this one file only.
  da3-base      Apache-2.0: the HF card (fetched) and the official README table (Depth-Anything-3 @ 3d835ec) agree.
  moge3         MIT: the HF card (in the snapshot); MoGe code MIT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_weights as fw  # noqa: E402  sha256, fetch (HF), write_manifest, verify: unchanged
import fetch_weights_da3 as fda3  # noqa: E402

ROMA_CODE = '77f8d68803526dcddfd9b7a46bc76125bdc25f15'  # = modal_apps/geometry_clean_ab.py ROMA_CODE (romatch 0.1.2)
FILES = {  # rel -> (primary URL, sha256, bytes); the URLs are romatch.models.model_zoo.weight_urls at ROMA_CODE
    'roma_outdoor': dict(path='roma/outdoor', code_revision=ROMA_CODE, files={
        'roma_outdoor.pth': ('https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth',
                             'c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba', 445647516),
        'LICENSE': ('https://raw.githubusercontent.com/Parskatt/storage/6fed4401e851d3c73ceaf15b885fea078a2701f1/LICENSE',  # = tag 'roma'
                    '73c934424d4489a3bf557902a07c693296484b3a3e4c3599003400ef15ae267c', 1070)},
        licence='MIT, inferred from the Parskatt/storage repository LICENSE (the release asset carries none); RoMa code MIT; MegaDepth-trained'),
    'roma_dinov2': dict(path='roma/dinov2', code_revision=ROMA_CODE, files={
        'dinov2_vitl14_pretrain.pth': ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
                                       'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428', 1217586395),
        'LICENSE': ('https://raw.githubusercontent.com/facebookresearch/dinov2/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE',
                    '600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7', 11359)},
        licence='Apache-2.0 (DINOv2 code and model weights); mirror this file only, the same tree hosts FAIR-NC models'),
}
_da3 = fda3.MODELS['da3-base']
HF = {'da3-base': dict(_da3, allow=[*_da3['allow'], 'README.md'],  # the card travels with the weights
                       sha256={**_da3['sha256'], 'README.md': '5388563071b4361b11d6d811b354d454f26aed2adba634fcf4ee3775c8eda480'}),
      'moge3': fw.MODELS['moge3']}


def fetch_files(cache: Path, name: str, spec: dict) -> dict:
    """Each file: kept if already there with the pinned SHA-256, else downloaded to .part, hashed, and only then renamed."""
    started, base, files = time.monotonic(), cache / spec['path'], {}
    for rel, (url, digest, _) in spec['files'].items():
        target = base / rel
        if not (target.is_file() and fw.sha256(target) == digest):
            part = target.with_name(target.name + '.part')
            part.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, part)
            got = fw.sha256(part)
            if got != digest:
                part.unlink()
                raise ValueError(f'{name} {rel}: SHA-256 {got} is not the pinned {digest} ({url})')
            part.replace(target)
        files[rel] = dict(sha256=digest, bytes=target.stat().st_size)
    return dict(layout='file', path=spec['path'], snapshot=spec['path'], link=None, gated=False, code_revision=spec['code_revision'],
                licence=spec['licence'], urls={rel: f[0] for rel, f in spec['files'].items()}, source='primary URLs', files=files,
                bytes=sum(f['bytes'] for f in files.values()), seconds=round(time.monotonic() - started, 1))


def roma_state_dicts(cache: Path, device='cpu') -> dict:
    """The weights= / dinov2_weights= arguments of romatch roma_outdoor() from the mirrored files: each file is read once and its
    SHA-256 checked against FILES before anything is unpickled (torch.load weights_only); a mismatch is refused. No download."""
    import hashlib
    import io
    blobs = {}
    for arg, name, rel in (('weights', 'roma_outdoor', 'roma_outdoor.pth'), ('dinov2_weights', 'roma_dinov2', 'dinov2_vitl14_pretrain.pth')):
        path = Path(cache) / FILES[name]['path'] / rel; data = path.read_bytes(); want = FILES[name]['files'][rel][1]
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise ValueError(f'{path}: SHA-256 {got} is not the pinned {want}; refusing to load it')
        blobs[arg] = data
    import torch
    return {arg: torch.load(io.BytesIO(data), map_location=device, weights_only=True) for arg, data in blobs.items()}


def fetch(cache: Path, name: str) -> dict:
    return fetch_files(cache, name, FILES[name]) if name in FILES else fw.fetch(cache, name, 'hub', HF[name])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--models', default='roma_outdoor,roma_dinov2,da3-base,moge3')
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
        got[name] = fetch(cache, name)
        print(json.dumps({name: {k: got[name].get(k) for k in ('revision', 'code_revision', 'bytes', 'seconds', 'licence')}}), flush=True)
    fw.write_manifest(cache, got)
    print(json.dumps({'manifest': str(cache / 'manifest.json'), 'models': sorted(got)}))


def _check():
    """URLs = romatch's own (when importable); a fake small file from a file:// URL lands at its path, enters the shared manifest
    and passes verify; a file already there is not re-downloaded; a wrong hash is refused and leaves nothing behind; a changed
    file is caught by verify."""
    import hashlib
    import tempfile
    try:
        from romatch.models.model_zoo import weight_urls
        assert weight_urls['romatch']['outdoor'] == FILES['roma_outdoor']['files']['roma_outdoor.pth'][0], weight_urls
        assert weight_urls['dinov2'] == FILES['roma_dinov2']['files']['dinov2_vitl14_pretrain.pth'][0], weight_urls
        urls = 'checked against romatch'
    except ImportError:
        urls = 'romatch not importable, URL check skipped'
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); data = b'weights' * 999; src = tmp / 'w.pth'; src.write_bytes(data)
        pin = (src.as_uri(), hashlib.sha256(data).hexdigest(), len(data))
        spec = dict(FILES['roma_outdoor'], files={'roma_outdoor.pth': pin})
        cache = tmp / 'cache'
        fw.write_manifest(cache, {'roma_outdoor': fetch_files(cache, 'roma_outdoor', spec)})
        assert (cache / 'roma/outdoor/roma_outdoor.pth').read_bytes() == data
        assert fw.verify(cache) == {'roma_outdoor': dict(files=1, bytes=len(data))}
        fetch_files(cache, 'roma_outdoor', dict(spec, files={'roma_outdoor.pth': ((tmp / 'gone').as_uri(), *pin[1:])}))  # no download
        try:
            fetch_files(tmp / 'c2', 'roma_outdoor', dict(spec, files={'roma_outdoor.pth': (pin[0], '0' * 64, pin[2])}))
            raise AssertionError('wrong hash accepted')
        except ValueError:
            pass
        assert not [p for p in (tmp / 'c2').rglob('*') if p.is_file()], 'a refused download left a file'
        (cache / 'roma/outdoor/roma_outdoor.pth').write_bytes(b'tampered')
        try:
            fw.verify(cache); raise AssertionError('changed file accepted')
        except ValueError:
            pass
        try:  # the run-time loader refuses the changed file too, before torch.load
            roma_state_dicts(cache); raise AssertionError('changed weights loaded')
        except ValueError:
            pass
    print(f'fetch_weights_geometry self-test passed: URLs {urls}; file:// fetch -> manifest -> verify; existing file kept; '
          'wrong hash refused with nothing left; changed file caught by verify and by roma_state_dicts')


if __name__ == '__main__':
    main()
