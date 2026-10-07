"""Fetch every pinned model revision of the workcell photo pipeline once into a local cache, SHA-256 checked, with a manifest.

  python scripts/onprem/fetch_weights.py --cache DIR [--models pi3x,sam3,moge3,recgen] [--source hub|HF_CACHE_DIR]
  python scripts/onprem/fetch_weights.py --cache DIR --verify          re-hash every file against DIR/manifest.json (offline)
  python scripts/onprem/fetch_weights.py --self-test

The only step that needs the internet (run it once, on any machine, then copy DIR). At run time stages use HF_HUB_OFFLINE=1 and
scripts/onprem/run_stage.py --weights DIR links each model to the path its stage reads:
  pi3x   -> /tmp/pi3x                   (modal_apps/pi3x_geometry.py downloads into that local_dir)
  sam3   -> /v/sam3/huggingface/hub     (workcell_sam_worker.py / workcell_mask_transfer.py cache_dir)
  moge3  -> HF_HOME=DIR/hf              (workcell_moge_check.py: MoGeModel.from_pretrained)
  recgen -> /cache                      (serving modal_apps/lucida_assets.py; also docker run -v DIR/recgen:/cache)
facebook/sam3 is gated: accept its licence on Hugging Face and export HF_TOKEN for this one fetch, or pass --source with an
existing Hugging Face cache that holds the pinned revision (e.g. the sam3-hf-cache volume). Integrity: every file is checked
against its Hugging Face content address (LFS SHA-256, or the git blob SHA-1 for small files) and the pins below.
The images set HF_HUB_OFFLINE=1; a fetch sets HF_HUB_OFFLINE=0 for this process only (--verify stays offline).
recgen writes DIR/recgen = serving lucida_assets.prepare_weights' /cache layout (weights/hf hub snapshot, weights/dinov2_*.pth,
recgen-weights-manifest.json with /cache paths, byte-identical to the one the reports used); --verify re-hashes all three.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

MODELS = {
    'pi3x': dict(repo='yyfz233/Pi3X', revision='bb1deea4d7423de5b30691739cb451a3f57dc1d5', layout='local', path='local/pi3x',
                 link='/tmp/pi3x', allow=['model.safetensors'],
                 sha256={'model.safetensors': '69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a'},
                 licence='weights CC-BY-NC-4.0 (non-commercial); code yyfz/Pi3 9fa3ddb BSD-style'),
    'sam3': dict(repo='facebook/sam3', revision='3c879f39826c281e95690f02c7821c4de09afae7', layout='hf', path='hf/hub',
                 link='/v/sam3/huggingface/hub', gated=True, ignore=['sam3.pt'],  # transformers reads model.safetensors
                 sha256={'model.safetensors': '6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a'},
                 licence='SAM License (Meta, custom; read before commercial use)'),
    'moge3': dict(repo='Ruicheng/moge-3-vitl', revision='184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be', layout='hf', path='hf/hub',
                  sha256={'model.pt': '9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925'}, licence='MIT'),
    'recgen': dict(repo='TRI-ML/RecGen', revision='bc0df7de2e43314830039a35a720731d4c4fac65', layout='recgen', path='recgen',
                   link='/cache', hf='recgen/weights/hf',
                   allow=['pipeline.json', 'ckpts/*', 'README.md', 'sparse-structure-ft-70k/*', 'slat_denoiser_ema0.9999_step0075000.pt',
                          'slat_config.json', 'slat_pose_stats.json'],  # = lucida_assets.prepare_weights
                   sha256={  # = lucida-replica-01 generation/environment-recgen/weights-manifest.json (the reports' weights)
                       'pipeline.json': 'c853cfe166c075c8431d9cd62a1418407950f220f007d4e805dc5f36c0ab1a2c',
                       'ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors': '38c84bcef5ce0af1f48b1b5558dabc7575a13346043c41a7e0610f1fa619a161',
                       'ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors': '3e87aba94b5786407eb06d0502c1ed0885a0027a3f2b8537bfe15b0a92c01859',
                       'ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors': '737da6578d01948016b7c39786113af0d64a46f7922f6b8b5e698b84643be514',
                       'ckpts/ss_dec_conv3d_16l8_fp16.safetensors': '1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a',
                       'slat_denoiser_ema0.9999_step0075000.pt': '30e1a77ff0112bc44b4776d9581104a6540863e2cf8a3b6281e73c11c8a67e9d',
                       'sparse-structure-ft-70k/stereo_denoiser_ema0.9999_step0070000.pt': '34ab3d26705822b7dcf157589d3b5143acd2a5972179deb20ad821baf587c4e7'},
                   extra=dict(name='dinov2_vitl14_reg4_pretrain.pth', path='recgen/weights/dinov2_vitl14_reg4_pretrain.pth',
                              url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
                              sha256='36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51',
                              code_revision='7764ea0f912e53c92e82eb78a2a1631e92725fc8'),
                   licence='weights CC-BY-NC-4.0, code Toyota Research Institute non-commercial; TRELLIS MIT; DINOv2 Apache-2.0'),
}
RECGEN_CODE_REV = 'fe3c9315b439c50ada8b60c12b469d739fd722db'  # serving modal_apps/lucida_assets.py CODE_REV


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def check_content_address(blob: Path) -> str:
    """A Hugging Face cache blob is named by its content: LFS SHA-256 (64 hex) or git blob SHA-1 (40 hex). Returns SHA-256."""
    digest = sha256(blob)
    name = blob.name
    if len(name) == 64 and digest != name or len(name) == 40 and git_blob_sha1(blob) != name or len(name) not in (40, 64):
        raise ValueError(f'{blob} does not match its content address')
    return digest


def repo_dir(hub: Path, repo: str) -> Path:
    return hub / ('models--' + repo.replace('/', '--'))


def copy_snapshot(source_hub: Path, hub: Path, repo: str, revision: str, ignore=()):
    """Copy one pinned snapshot (its blobs, snapshot links, refs and no-exist markers) between Hugging Face caches."""
    src, dst = repo_dir(source_hub, repo), repo_dir(hub, repo)
    snap = src / 'snapshots' / revision
    if not snap.is_dir():
        raise FileNotFoundError(f'{snap} missing: the source cache does not hold {repo}@{revision}')
    for link in sorted(p for p in snap.rglob('*') if p.is_symlink() or p.is_file()):
        rel = link.relative_to(snap)
        if any(rel.match(pattern) for pattern in ignore):
            continue
        blob = link.resolve()
        (dst / 'blobs').mkdir(parents=True, exist_ok=True)
        if not (dst / 'blobs' / blob.name).exists():
            shutil.copyfile(blob, dst / 'blobs' / (blob.name + '.part')); (dst / 'blobs' / (blob.name + '.part')).rename(dst / 'blobs' / blob.name)
        out = dst / 'snapshots' / revision / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.symlink_to(os.path.relpath(dst / 'blobs' / blob.name, out.parent))
    for extra in ('refs', f'.no_exist/{revision}'):
        if (src / extra).exists():
            shutil.copytree(src / extra, dst / extra, dirs_exist_ok=True)


def hub_snapshot(hub: Path, spec: dict, source):
    if source == 'hub':
        from huggingface_hub import snapshot_download  # HF_TOKEN from the environment for gated repos; never printed
        snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=str(hub), allow_patterns=spec.get('allow'),
                          ignore_patterns=spec.get('ignore'))
    else:
        copy_snapshot(Path(source), hub, spec['repo'], spec['revision'], spec.get('ignore', ()))
    return repo_dir(hub, spec['repo']) / 'snapshots' / spec['revision']


def verify_files(snap: Path, spec: dict) -> dict:
    files = {}
    for p in sorted(x for x in snap.rglob('*') if x.is_file()):
        rel = str(p.relative_to(snap))
        digest = check_content_address(p.resolve()) if p.is_symlink() else sha256(p)
        if rel in spec.get('sha256', {}) and digest != spec['sha256'][rel]:
            raise ValueError(f"{spec['repo']} {rel}: SHA-256 {digest} is not the pinned {spec['sha256'][rel]}")
        files[rel] = dict(sha256=digest, bytes=p.stat().st_size)
    missing = set(spec.get('sha256', {})) - set(files)
    if missing:
        raise ValueError(f"{spec['repo']}: pinned files missing {sorted(missing)}")
    return files


def fetch(cache: Path, name: str, source: str, spec: dict | None = None) -> dict:
    spec = spec or MODELS[name]
    # The images default to HF_HUB_OFFLINE=1; huggingface_hub reads it when first imported (lazily, below), so this wins.
    os.environ['HF_HUB_OFFLINE'] = '0'
    started = time.monotonic()
    extras = {}
    if spec['layout'] == 'local':  # exactly the stage's own hf_hub_download(local_dir=...), so its metadata matches the pin
        from huggingface_hub import hf_hub_download
        target = cache / spec['path']
        for filename in spec['allow']:
            hf_hub_download(spec['repo'], filename, revision=spec['revision'], local_dir=str(target))
        files = {f: dict(sha256=sha256(target / f), bytes=(target / f).stat().st_size) for f in spec['allow']}
        for f, row in files.items():
            if row['sha256'] != spec['sha256'][f]:
                raise ValueError(f'{name} {f}: SHA-256 {row["sha256"]} is not the pinned {spec["sha256"][f]}')
        snapshot = target
    elif spec['layout'] == 'hf':
        snapshot = hub_snapshot(cache / spec['path'], spec, source)
        files = verify_files(snapshot, spec)
    else:  # recgen: lucida_assets.prepare_weights' layout under DIR/recgen, mounted at /cache
        snapshot = hub_snapshot(cache / spec['hf'], spec, source)
        files = verify_files(snapshot, spec)
        extra = spec['extra']; dino = cache / extra['path']
        if not dino.is_file() or sha256(dino) != extra['sha256']:
            dino.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(extra['url'], dino.with_suffix('.part')); dino.with_suffix('.part').rename(dino)
        if sha256(dino) != extra['sha256']:
            raise ValueError('DINOv2 checkpoint SHA-256 mismatch')
        root = '/cache'  # container paths, as lucida_assets.generate_object reads them
        stage = dict(model_id=spec['repo'], model_revision=spec['revision'], code_revision=RECGEN_CODE_REV,
                     snapshot=f"{root}/{snapshot.relative_to(cache / spec['path'])}", files=files,
                     dino=dict(path=f"{root}/{dino.relative_to(cache / spec['path'])}", url=extra['url'], sha256=extra['sha256'],
                               bytes=dino.stat().st_size, code_revision=extra['code_revision']),
                     licenses={'recgen_code': 'Toyota Research Institute Non-Commercial', 'recgen_weights': 'CC-BY-NC-4.0',
                               'trellis_base': 'MIT', 'dinov2': 'Apache-2.0', 'flexicubes': 'Apache-2.0'})
        stage_manifest = cache / spec['path'] / 'recgen-weights-manifest.json'
        stage_manifest.write_text(json.dumps(stage, indent=2))  # same bytes as prepare_weights writes
        extras = {extra['name']: str(dino.relative_to(cache)), stage_manifest.name: str(stage_manifest.relative_to(cache))}
        files = {**files, extra['name']: dict(sha256=extra['sha256'], bytes=dino.stat().st_size),
                 stage_manifest.name: dict(sha256=sha256(stage_manifest), bytes=stage_manifest.stat().st_size)}
    entry = {k: spec[k] for k in ('repo', 'revision', 'layout', 'path', 'licence') if k in spec}
    entry.update(link=spec.get('link'), gated=bool(spec.get('gated')), source='huggingface.co' if source == 'hub' else 'local cache copy',
                 snapshot=str(snapshot.relative_to(cache)), files=files, bytes=sum(f['bytes'] for f in files.values()),
                 **({'extras': extras} if extras else {}),
                 seconds=round(time.monotonic() - started, 1))
    return entry


def write_manifest(cache: Path, models: dict):
    path = cache / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else dict(schemaVersion=1, hfHome='hf', models={})
    manifest['models'].update(models)
    manifest['updated'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


def verify(cache: Path) -> dict:
    manifest = json.loads((cache / 'manifest.json').read_text())
    out = {}
    for name, m in manifest['models'].items():
        base, extras = cache / m['snapshot'], m.get('extras', {})  # extras: files outside the snapshot (recgen DINOv2, manifest)
        for rel, row in m['files'].items():
            p = cache / extras[rel] if rel in extras else base / rel
            if not p.is_file():
                raise FileNotFoundError(f'{name} {rel}: missing ({p})')
            if sha256(p) != row['sha256']:
                raise ValueError(f'{name} {rel}: SHA-256 changed since the fetch')
        out[name] = dict(files=len(m['files']), bytes=m['bytes'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--models', default='pi3x,sam3,moge3,recgen')
    ap.add_argument('--source', default='hub', help="'hub' (huggingface.co) or an existing Hugging Face cache dir (…/hub)")
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return _check()
    cache = args.cache.resolve(); cache.mkdir(parents=True, exist_ok=True)
    if args.verify:
        print(json.dumps(verify(cache)))
        return
    got = {}
    for name in args.models.split(','):
        got[name] = fetch(cache, name, args.source)
        print(json.dumps({name: {k: got[name][k] for k in ('revision', 'bytes', 'seconds', 'source')}}), flush=True)
    write_manifest(cache, got)
    print(json.dumps({'manifest': str(cache / 'manifest.json'), 'models': sorted(got)}))


def _check():
    """A fake Hugging Face cache (one LFS blob, one git blob) copies, verifies, rejects a corrupted blob and a wrong pin."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); src = repo_dir(tmp / 'src', 'org/m'); rev = 'a' * 40
        big, small = b'weights' * 1000, b'{"x": 1}'
        names = {'model.bin': hashlib.sha256(big).hexdigest(), 'config.json': hashlib.sha1(b'blob %d\0' % len(small) + small).hexdigest()}
        (src / 'blobs').mkdir(parents=True); (src / 'snapshots' / rev).mkdir(parents=True); (src / 'refs').mkdir()
        (src / 'refs/main').write_text(rev)
        for f, data in (('model.bin', big), ('config.json', small)):
            (src / 'blobs' / names[f]).write_bytes(data); (src / 'snapshots' / rev / f).symlink_to(f'../../blobs/{names[f]}')
        spec = dict(repo='org/m', revision=rev, sha256={'model.bin': names['model.bin']})
        snap = hub_snapshot(tmp / 'dst', spec, str(tmp / 'src'))
        files = verify_files(snap, spec)
        assert files['model.bin']['sha256'] == names['model.bin'] and files['config.json']['bytes'] == len(small), files
        try:
            verify_files(snap, {**spec, 'sha256': {'model.bin': '0' * 64}}); raise AssertionError('wrong pin accepted')
        except ValueError:
            pass
        (repo_dir(tmp / 'dst', 'org/m') / 'blobs' / names['model.bin']).write_bytes(b'tampered')
        try:
            verify_files(snap, spec); raise AssertionError('corrupted blob accepted')
        except ValueError:
            pass
        # recgen layout from the same fake cache + a DINOv2 file (file:// URL), with HF_HUB_OFFLINE=1 as in the images
        dino = b'dino' * 999; (tmp / 'dino.pth').write_bytes(dino)
        rec = dict(MODELS['recgen'], repo='org/m', revision=rev, sha256={'model.bin': names['model.bin']},
                   extra=dict(MODELS['recgen']['extra'], url=(tmp / 'dino.pth').as_uri(), sha256=hashlib.sha256(dino).hexdigest()))
        os.environ['HF_HUB_OFFLINE'] = '1'
        cache = tmp / 'cache'
        write_manifest(cache, {'recgen': fetch(cache, 'recgen', str(tmp / 'src'), rec)})
        assert os.environ['HF_HUB_OFFLINE'] == '0', "fetch must override the images' HF_HUB_OFFLINE=1"
        stage = json.loads((cache / 'recgen/recgen-weights-manifest.json').read_text())
        assert stage['snapshot'] == f'/cache/weights/hf/models--org--m/snapshots/{rev}' and set(stage['files']) == {'model.bin', 'config.json'}, stage
        assert stage['dino']['path'] == '/cache/weights/dinov2_vitl14_reg4_pretrain.pth' and stage['dino']['sha256'] == rec['extra']['sha256']
        assert verify(cache)['recgen']['files'] == 4  # snapshot (2) + DINOv2 + recgen-weights-manifest.json
        for victim, error in ((cache / 'recgen/weights/dinov2_vitl14_reg4_pretrain.pth', ValueError),
                              (cache / 'recgen/recgen-weights-manifest.json', FileNotFoundError)):
            keep = victim.read_bytes()
            victim.write_bytes(b'x') if error is ValueError else victim.unlink()
            try:
                verify(cache); raise AssertionError(f'verify accepted a changed {victim.name}')
            except error:
                pass
            victim.write_bytes(keep)
        # a non-recgen model with a missing file: a clear FileNotFoundError, not KeyError('extra')
        write_manifest(cache, {'m': fetch(cache, 'm', str(tmp / 'src'), dict(spec, layout='hf', path='hf/hub'))})
        (cache / 'hf/hub/models--org--m/snapshots' / rev / 'config.json').unlink()
        try:
            verify(cache); raise AssertionError('missing file accepted')
        except FileNotFoundError:
            pass
    print('fetch_weights self-test passed: snapshot copy, content-address and pin checks; recgen layout (snapshot + DINOv2 + '
          'manifest) with HF_HUB_OFFLINE=1 overridden; verify catches a changed DINOv2, a missing manifest, a missing file')


if __name__ == '__main__':
    main()
