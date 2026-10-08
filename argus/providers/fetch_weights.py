"""SHA-256 verification and pinned Hugging Face cache fetching for the retained geometry and SAM3D models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

MODELS = {'moge3': dict(repo='Ruicheng/moge-3-vitl', revision='184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be', layout='hf', path='hf/hub',
                  sha256={'model.pt': '9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925'}, licence='MIT')}


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
    else:
        raise ValueError(f"Unsupported weight layout: {spec['layout']}")
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
        base, extras = cache / m['snapshot'], m.get('extras', {})  # files outside the snapshot, if recorded
        for rel, row in m['files'].items():
            p = cache / extras[rel] if rel in extras else base / rel
            if not p.is_file():
                raise FileNotFoundError(f'{name} {rel}: missing ({p})')
            if sha256(p) != row['sha256']:
                raise ValueError(f'{name} {rel}: SHA-256 changed since the fetch')
        out[name] = dict(files=len(m['files']), bytes=m['bytes'])
    return out
