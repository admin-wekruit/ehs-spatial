"""Compact collection must honor byte limits, available space and content hashes."""
import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modal_apps'))
import lingbot_cloud_compare as compare

payload = b'original remote preview'
record = {'name': 'preview.glb', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
with TemporaryDirectory() as temporary:
    root = Path(temporary)
    volume = SimpleNamespace(read_file=lambda _: iter([payload[:5], payload[5:]]))
    with patch.object(compare, 'volume', volume), patch.object(compare.shutil, 'disk_usage', return_value=SimpleNamespace(free=3*1024**3)):
        compare.collect_preview('saved-run', root, {'files': [record]})
        assert (root/'preview.glb').read_bytes() == payload
        explicit = {**record, 'name': 'other-volume.glb'}
        with patch.object(compare, 'volume', None):
            compare.collect_preview('runs/saved-run', root, {'files': [explicit]}, artifact_volume=volume)
        assert (root/explicit['name']).read_bytes() == payload
        for bad in [{**record, 'name': '../escape'}, {**record, 'name': 'oversize', 'bytes': len(payload)-1},
                    {**record, 'name': 'bad-hash', 'sha256': '0'*64}]:
            try: compare.collect_preview('saved-run', root, {'files': [bad]})
            except (ValueError, AssertionError): pass
            else: raise AssertionError('Unbound preview accepted')
    for free, size in [(2*1024**3, len(payload)), (4*1024**3, 32*1024**2+1)]:
        with patch.object(compare.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)):
            try: compare.collect_preview('saved-run', root, {'files': [{**record, 'bytes': size}]})
            except OSError: pass
            else: raise AssertionError('Collection resource limit ignored')
print('PASS: bounded verified preview collection without full prediction downloads')
