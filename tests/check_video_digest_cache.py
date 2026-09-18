"""Run directly: reuse unchanged-file hashes, never stale or partial results."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import reconstruct_room_rgb as files

original = hashlib.file_digest
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'model.bin'
    path.write_bytes(b'a' * 4096)
    link = Path(folder) / 'alias.bin'
    link.symlink_to(path)
    with patch.object(files.hashlib, 'file_digest', wraps=original) as hash_file:
        expected = hashlib.sha256(b'a' * 4096).hexdigest()
        for source in [path, str(path), link] * 10:
            assert files.digest(source) == expected
        assert hash_file.call_count == 1

        before = path.stat()
        path.write_bytes(b'b' * 4096)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert path.stat().st_ctime_ns != before.st_ctime_ns
        assert files.digest(path) == hashlib.sha256(b'b' * 4096).hexdigest()
        assert hash_file.call_count == 2, 'Same-size edits with restored mtime must invalidate'

        replacement = Path(folder) / 'replacement.bin'
        replacement.write_bytes(b'c' * 4096)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
        assert files.digest(path) == hashlib.sha256(b'c' * 4096).hexdigest()
        assert hash_file.call_count == 3, 'Atomic replacements must invalidate'

    files._file_digest.cache_clear()
    def change_during_read(stream, algorithm):
        value = original(stream, algorithm)
        path.write_bytes(b'd' * 4096)
        return value
    with patch.object(files.hashlib, 'file_digest', side_effect=change_during_read):
        try:
            files.digest(path)
        except OSError as error:
            assert 'changed while hashing' in str(error)
        else:
            raise AssertionError('Never cache a file that changed during hashing')
    assert files._file_digest.cache_info().currsize == 0
    assert files.digest(path) == hashlib.sha256(b'd' * 4096).hexdigest()
    path.unlink()
    try:
        files.digest(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('A cached digest cannot conceal a missing file')
assert files._file_digest.cache_parameters()['maxsize'] == 4096
print('PASS: 30 unchanged reads hash once; edits, replacements and missing files invalidate; concurrent changes reject; bounded metadata-only cache')
