"""Split the immutable run binary into exact, lossless per-object downloads."""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import sys

source = Path(sys.argv[1])
root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent
scene = json.loads((root / 'scene.json').read_text())
original = json.loads((source / 'scene.json').read_text())
binary = (source / 'scene.bin').read_bytes()
scene['source_binary'] = {'bytes': len(binary), 'sha256': hashlib.sha256(binary).hexdigest()}
objects = {o['id']: o for o in original['objects']}
assert set(objects) == {o['id'] for o in scene['objects']}
reconstructed = []
for obj in scene['objects']:
    mesh = objects[obj['id']]['mesh']
    start = mesh['byte_offset']
    end = mesh['index_byte_offset'] + mesh['index_count'] * 4
    raw = binary[start:end]
    minimum = [float('inf')] * 3
    maximum = [-float('inf')] * 3
    for vertex in struct.iter_unpack('<9f', raw[:mesh['vertex_count'] * 36]):
        for axis in range(3):
            minimum[axis] = min(minimum[axis], vertex[axis])
            maximum[axis] = max(maximum[axis], vertex[axis])
    reconstructed.append(raw)
    packed = gzip.compress(raw, compresslevel=6, mtime=0)
    target = root / 'model' / 'objects' / (obj['id'] + '.bin.gz')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(packed)
    obj['mesh'] = dict(mesh, byte_offset=0, index_byte_offset=mesh['index_byte_offset'] - start,
                       bounds={'min': minimum, 'max': maximum},
                       asset={'path': str(target.relative_to(root)), 'bytes': len(raw),
                              'packed_bytes': len(packed), 'sha256': hashlib.sha256(raw).hexdigest()})
    assert gzip.decompress(packed) == raw
    assert obj['mesh']['vertex_count'] == mesh['vertex_count']
    assert obj['mesh']['index_count'] == mesh['index_count']
assert b''.join(reconstructed) == binary
scene.pop('binary', None)
scene.pop('binary_asset', None)
(root / 'scene.json').write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'objects': len(scene['objects']), 'download_bytes': sum(o['mesh']['asset']['packed_bytes'] for o in scene['objects']),
                  'original_bytes': len(binary), 'triangles': sum(o['mesh']['index_count'] // 3 for o in scene['objects']),
                  'per_object_bytes_identical': True}))
