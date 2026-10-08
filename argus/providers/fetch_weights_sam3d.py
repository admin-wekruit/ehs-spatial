"""Fetch the weights of docker/sam3d.Dockerfile once into the same local cache as fetch_weights.py (one shared manifest.json):
SAM 3D Objects at its pinned revision, mesh-only files only, and the DINOv2 ViT-L/14 reg4 checkpoint its embedders load
through torch.hub. SHA-256 checked; the image never holds weights.

  python scripts/onprem/fetch_weights_sam3d.py --cache DIR [--source hub|HF_CACHE_DIR] [--dinov2 URL|FILE]
  python scripts/onprem/fetch_weights_sam3d.py --cache DIR --verify          re-hash every file of DIR/manifest.json (offline)
  python scripts/onprem/fetch_weights_sam3d.py --self-test

What lands where (stages then run with scripts/onprem/run_stage.py --weights DIR, HF_HUB_OFFLINE=1, no network):
  sam3d         DIR/hf/hub/models--facebook--sam-3d-objects/snapshots/<rev>: checkpoints/pipeline.yaml and the ss_generator,
                slat_generator, ss_decoder and slat_decoder_mesh configs + checkpoints (12.1 GB), LICENSE, README.md.
                run_stage sets HF_HOME=DIR/hf, so sam3d_research.SAM3DObjects.load's snapshot_download(revision=...) finds it
                offline. Not fetched: the Gaussian decoders, the encoders, slat_decoder_mesh.pt (unused duplicate), doc/.
  sam3d_dinov2  DIR/sam3d/torch-hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth; run_stage links /opt/torch-hub/hub/checkpoints
                (torch.hub's checkpoint dir under the image's TORCH_HOME, next to the vendored DINOv2 code) to that directory.
facebook/sam-3d-objects is gated: accept its licence on Hugging Face and export HF_TOKEN for this one fetch, or pass --source
with an existing Hugging Face cache that holds the pinned revision (e.g. the panoptes-sam3d-weights volume's huggingface/hub).
The SAM License must travel with the weights: LICENSE is part of the fetched set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

import argus.providers.fetch_weights as fw

REVISION = '2e73555018d2741ccd486e56c24fac41155a1dc6'  # = modal_apps/sam3d_research.py MODEL_REVISION
MESH_ONLY = ['LICENSE', 'README.md', 'checkpoints/pipeline.yaml'] + [
    f'checkpoints/{m}.{e}' for m in ('ss_generator', 'slat_generator', 'ss_decoder', 'slat_decoder_mesh') for e in ('yaml', 'ckpt')]
SAM3D = dict(repo='facebook/sam-3d-objects', revision=REVISION, layout='hf', path='hf/hub', gated=True, allow=MESH_ONLY,
             # the same selection for a local-cache copy (fetch_weights.copy_snapshot only knows ignore patterns)
             ignore=['.gitattributes', 'CODE_OF_CONDUCT.md', 'CONTRIBUTING.md', 'doc/*', 'checkpoints/slat_decoder_gs*',
                     'checkpoints/slat_decoder_mesh.pt', 'checkpoints/slat_encoder*', 'checkpoints/ss_encoder*'],
             sha256={  # LFS SHA-256 at REVISION: the content-addressed blob names of the HF cache the A/B used
                 # (panoptes-sam3d-weights; sizes = the HF tree API's, which hides the hashes of this gated repo)
                 'checkpoints/ss_generator.ckpt': '225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146',
                 'checkpoints/slat_generator.ckpt': '91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb',
                 'checkpoints/ss_decoder.ckpt': '6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340',
                 'checkpoints/slat_decoder_mesh.ckpt': '85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33',
             },
             licence='SAM License (Meta, custom; no ITAR/military/nuclear/espionage use; redistribute with the licence text)')
DINOV2 = dict(name='dinov2_vitl14_reg4_pretrain.pth', path='sam3d/torch-hub/checkpoints', link='/opt/torch-hub/hub/checkpoints',
              url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
              sha256='36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51',  # = original shared DINOv2 checkpoint
              code_revision='7764ea0f912e53c92e82eb78a2a1631e92725fc8',  # vendored in docker/sam3d.Dockerfile
              licence='Apache-2.0 (DINOv2 code and weights)')


def fetch_sam3d(cache: Path, source: str, spec: dict = SAM3D) -> dict:
    entry = fw.fetch(cache, 'sam3d', source, spec)
    got = set(entry['files'])
    if got != set(spec['allow']):  # a missing config, or a file the mesh-only set must not carry
        raise ValueError(f"sam3d: snapshot files {sorted(got ^ set(spec['allow']))} differ from the mesh-only set")
    return entry


def fetch_dinov2(cache: Path, source: str, spec: dict = DINOV2) -> dict:
    started = time.monotonic()
    target = cache / spec['path'] / spec['name']
    if not target.is_file() or fw.sha256(target) != spec['sha256']:
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix('.part')
        if Path(source).is_file():
            shutil.copyfile(source, part)
        else:
            urllib.request.urlretrieve(source, part)
        part.rename(target)
    if fw.sha256(target) != spec['sha256']:
        raise ValueError('DINOv2 checkpoint SHA-256 mismatch')
    files = {spec['name']: dict(sha256=spec['sha256'], bytes=target.stat().st_size)}
    return dict(url=spec['url'], code_revision=spec['code_revision'], layout='file', path=spec['path'], link=spec['link'],
                licence=spec['licence'], source='local file copy' if Path(source).is_file() else source, snapshot=spec['path'],
                files=files, bytes=files[spec['name']]['bytes'], gated=False, seconds=round(time.monotonic() - started, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--source', default='hub', help="SAM 3D: 'hub' (huggingface.co, HF_TOKEN) or an existing HF cache dir (…/hub)")
    ap.add_argument('--dinov2', default=DINOV2['url'], help='DINOv2 checkpoint: its URL (default) or a local copy')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return _check()
    cache = args.cache.resolve(); cache.mkdir(parents=True, exist_ok=True)
    if args.verify:
        print(json.dumps(fw.verify(cache)))
        return
    got = {'sam3d': fetch_sam3d(cache, args.source), 'sam3d_dinov2': fetch_dinov2(cache, args.dinov2)}
    for name, entry in got.items():
        print(json.dumps({name: {k: entry.get(k) for k in ('revision', 'code_revision', 'bytes', 'seconds', 'source')}}), flush=True)
    fw.write_manifest(cache, got)
    print(json.dumps({'manifest': str(cache / 'manifest.json'), 'models': sorted(got)}))


def _check():
    """A fake gated repo cache (wanted + unwanted files) copies to exactly the wanted set; a missing config is refused; the
    DINOv2 file (local copy) lands where run_stage links it, verify re-hashes both, and a tampered checkpoint is caught."""
    import hashlib
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); rev = 'b' * 40; src = fw.repo_dir(tmp / 'src', 'org/m'); snap = src / 'snapshots' / rev
        files = {'LICENSE': b'licence', 'checkpoints/a.yaml': b'a: 1', 'checkpoints/a.ckpt': b'w' * 5000,
                 'checkpoints/gs.ckpt': b'g' * 100, 'doc/x.png': b'png'}
        lfs = {'checkpoints/a.ckpt', 'checkpoints/gs.ckpt'}
        (src / 'blobs').mkdir(parents=True)
        for rel, data in files.items():
            name = hashlib.sha256(data).hexdigest() if rel in lfs else hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
            (src / 'blobs' / name).write_bytes(data)
            (snap / rel).parent.mkdir(parents=True, exist_ok=True); (snap / rel).symlink_to(os.path.relpath(src / 'blobs' / name, (snap / rel).parent))
        spec = dict(SAM3D, repo='org/m', revision=rev, allow=['LICENSE', 'checkpoints/a.yaml', 'checkpoints/a.ckpt'],
                    ignore=['checkpoints/gs*', 'doc/*'], sha256={'checkpoints/a.ckpt': hashlib.sha256(files['checkpoints/a.ckpt']).hexdigest()})
        cache = tmp / 'cache'
        sam = fetch_sam3d(cache, str(tmp / 'src'), spec)
        assert set(sam['files']) == set(spec['allow']) and sam['snapshot'] == f'hf/hub/models--org--m/snapshots/{rev}', sam
        try:
            fetch_sam3d(tmp / 'cache2', str(tmp / 'src'), dict(spec, allow=spec['allow'] + ['checkpoints/b.yaml'])); raise AssertionError('missing file accepted')
        except ValueError:
            pass
        dino = b'dino' * 777; (tmp / 'dino.pth').write_bytes(dino)
        dspec = dict(DINOV2, sha256=hashlib.sha256(dino).hexdigest(), link=str(tmp / 'image/torch-hub/hub/checkpoints'))
        d = fetch_dinov2(cache, str(tmp / 'dino.pth'), dspec)
        fw.write_manifest(cache, {'sam3d': sam, 'sam3d_dinov2': d})
        assert fw.verify(cache) == {'sam3d': dict(files=3, bytes=sum(len(files[f]) for f in spec['allow'])), 'sam3d_dinov2': dict(files=1, bytes=len(dino))}
        try:
            fetch_dinov2(tmp / 'cache3', str(tmp / 'dino.pth'), dict(dspec, sha256='0' * 64)); raise AssertionError('wrong DINOv2 accepted')
        except ValueError:
            pass
        (cache / dspec['path'] / DINOV2['name']).write_bytes(b'tampered')
        try:
            fw.verify(cache); raise AssertionError('tampered DINOv2 accepted')
        except ValueError:
            pass
    print('fetch_weights_sam3d self-test passed: mesh-only selection from a gated-repo cache (extra files left out, a missing '
          'one refused), DINOv2 copy + SHA-256, shared manifest verify, pinned torch.hub checkpoint path')


if __name__ == '__main__':
    main()
