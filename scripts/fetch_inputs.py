"""Fetch frozen inputs from an explicit file/HTTPS mirror or the existing S3 blob store, then verify bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote, unquote, urlsplit
from urllib.request import urlopen

from argus import ROOT


def fetch(manifest: Path, source: str, destination: Path) -> int:
    entries = json.loads(manifest.read_text())['files']
    root = destination.resolve()
    source_url = urlsplit(source)
    if source_url.scheme not in ('file', 'https', 'http', 's3'):
        raise ValueError('input source must be file://, https://, http:// or s3://')
    if source_url.scheme == 'file' and source_url.netloc not in ('', 'localhost'):
        raise ValueError('file mirror must be local')
    store = None
    if source_url.scheme == 's3':
        from argus.platform.s3_storage import S3BlobStore
        store = S3BlobStore.from_url(source)
    count = 0
    for entry in entries:
        relative = PurePosixPath(entry['path'])
        path = root / relative
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root):
            raise ValueError('manifest path escapes the data root')
        sha, size = entry['sha256'], entry['sizeBytes']
        if not re.fullmatch(r'[a-f0-9]{64}', sha) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError('invalid input digest or size')
        matches = lambda data: len(data) == size and hashlib.sha256(data).hexdigest() == sha
        if path.is_file() and matches(path.read_bytes()):
            continue
        if store:
            data = store.get('sha256/' + sha, sha, size)
        elif source_url.scheme == 'file':
            data = (Path(unquote(source_url.path)) / relative).read_bytes()
        else:
            with urlopen(source.rstrip('/') + '/' + quote(str(relative), safe='/'), timeout=180) as response:
                data = response.read()
        if not matches(data):
            raise ValueError(f'input integrity mismatch: {relative}')
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.input-', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data-manifest.json')
    parser.add_argument('--source', default=os.environ.get('PANOPTES_INPUT_SOURCE'))
    parser.add_argument('--dest', type=Path, default=Path(os.environ.get('PANOPTES_DATA_ROOT', ROOT / 'data')))
    args = parser.parse_args()
    if not args.source:
        parser.error('--source or PANOPTES_INPUT_SOURCE is required')
    print(f'inputs verified; {fetch(args.manifest, args.source, args.dest)} files fetched')


if __name__ == '__main__':
    main()
