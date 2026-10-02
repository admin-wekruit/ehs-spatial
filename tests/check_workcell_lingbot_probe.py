"""CPU check: crop coordinates, frozen observation binding, compact native mesh."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'modal_apps')]
from workcell_lingbot_probe import canonical_image, export_surface, frozen_inputs
from lingbot_room import saved_world_to_camera
import hashlib

# Independent known geometry: rotate camera +90 degrees about world Z, and
# place its optical center at (2,3,4). Camera point (1,2,5) is world (0,4,9).
c2w = np.array([[0., -1., 0., 2.], [1., 0., 0., 3.], [0., 0., 1., 4.]])
w2c = saved_world_to_camera(c2w)
assert np.allclose(w2c @ [0., 4., 9., 1.], [1., 2., 5.])
assert np.allclose(w2c @ [2., 3., 4., 1.], [0., 0., 0.])
assert not np.allclose(c2w @ [0., 4., 9., 1.], [1., 2., 5.])

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = Image.new('RGB', (3024, 4032), (42, 63, 84))
    source.save(root / 'portrait.png')
    rgb, affine, raw_wh = canonical_image(root / 'portrait.png')
    assert raw_wh == (3024, 4032) and rgb.shape == (518, 518, 3)
    assert np.array_equal(rgb[0, 0], [42, 63, 84])
    # Pixel-center convention, not zero-origin scaling: the raw image center
    # maps to the canonical center after the rounded-height crop.
    assert np.allclose(affine @ [1511.5, 2015.5, 1], [258.5, 258.5, 1])
    assert np.allclose(np.linalg.inv(affine) @ [258.5, 258.5, 1], [1511.5, 2015.5, 1])
    assert np.isclose(affine[1, 1], 686 / 4032)
    assert np.isclose(affine[1, 2], (686 / 4032 - 1) / 2 - 84)
    reference = {'features': {'wholeComponentHeightM': .1, 'mainBodyDiameterM': .085, 'redActuatorDiameterM': .04}}
    rows = [{'photo': p, 'redHullRaw': [[p, 2], [3, 4]]} for p in [2, 3, 4]]
    sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    (root / 'geometry.json').write_text(json.dumps({'anchor': {'referenceFit': {
        'observations': rows, 'sourceContourSha256': sha, 'reference': reference}}}))
    (root / 'geometry-timing.json').write_text(json.dumps({'frameSummaries': [
        {'sourceSha256': str(i)} for i in range(4)]}))
    measurements = root / 'measurements.json'
    measurements.write_text(json.dumps({'reference': reference, 'evaluation': {'fence': 999}}))
    first = frozen_inputs(root, measurements)
    measurements.write_text(json.dumps({'reference': reference, 'evaluation': {'fence': 1}}))
    assert frozen_inputs(root, measurements) == first
    assert 'evaluation' not in first and 'evaluation' not in first['reference']
    broken = json.loads((root / 'geometry.json').read_text())
    broken['anchor']['referenceFit']['observations'][0]['photo'] = 1
    (root / 'geometry.json').write_text(json.dumps(broken))
    try:
        frozen_inputs(root, measurements)
    except ValueError as exc:
        assert 'contours changed' in str(exc)
    else:
        raise AssertionError('Changed source contour passed its hash binding')
    yy, xx = np.indices((12, 12))
    points = np.stack([xx, yy, np.full_like(xx, 2)], -1).astype(float)
    export_surface({1: {'points': points, 'depth': points[..., 2],
                        'canonicalRgb': np.full((12, 12, 3), 123, np.uint8)}}, root)
    mesh = trimesh.load(root / 'observed-native.glb', force='scene')
    assert len(mesh.geometry) == 1
    only = next(iter(mesh.geometry.values()))
    assert len(only.faces) == 8 and np.isfinite(only.vertices).all()
    assert np.allclose(only.vertices[:, 2], 2.)
print('PASS: independent C2W/W2C geometry, exact crop centers, frozen observations/GT exclusion, native compact surface')
