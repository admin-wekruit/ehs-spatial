"""A bad mirror must not replace valid local bytes; manifest paths stay inside the selected root."""
import hashlib
import json
from pathlib import Path

import pytest
from scripts.fetch_inputs import fetch


def test_fetch_verified_bytes_and_preserve_existing_on_failure(tmp_path):
    mirror = tmp_path / 'mirror'; mirror.mkdir(); (mirror / 'photo.bin').write_bytes(b'photo')
    manifest = tmp_path / 'manifest.json'
    entry = dict(path='photo.bin', sha256=hashlib.sha256(b'photo').hexdigest(), sizeBytes=5)
    manifest.write_text(json.dumps(dict(files=[entry])))
    destination = tmp_path / 'data'
    assert fetch(manifest, mirror.as_uri(), destination) == 1
    assert fetch(manifest, mirror.as_uri(), destination) == 0
    (destination / 'photo.bin').write_bytes(b'keep')
    (mirror / 'photo.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity mismatch'):
        fetch(manifest, mirror.as_uri(), destination)
    assert (destination / 'photo.bin').read_bytes() == b'keep'
    assert list(destination.glob('.input-*')) == []
    entry['path'] = '../outside'
    manifest.write_text(json.dumps(dict(files=[entry])))
    with pytest.raises(ValueError, match='escapes'):
        fetch(manifest, mirror.as_uri(), destination)
