"""Check exported browser points against native depth support and original RGB.

Run: python tests/check_droid_display.py /path/to/replay/scene.json
"""
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

scene_path = Path(sys.argv[1])
scene = json.loads(scene_path.read_text())
assert scene.get('pointCloudUrl'), 'Browser cloud must preserve source RGB in a GLB'
rows = json.loads((scene_path.parent / 'geometry-frames.json').read_text())
with np.load(scene_path.parent / 'depth-support.npz') as data:
    retained = data['retained']
with np.load(Path(scene['provenance']['source_run']) / 'prediction.npz') as data:
    valid = data['keyframe_final_fullres_valid'][:, ::2, ::2]
    rgb = data['keyframe_model_bgr'][:, ::2, ::2, ::-1]
    allowed = np.concatenate([retained[i][valid[i]] for i in range(len(rows))])
    colors = np.concatenate([rgb[i][valid[i]] for i in range(len(rows))])
ids = np.array([p[0] for p in scene['points']], dtype=int)
assert 0 < len(ids) <= 300000 and np.all(np.diff(ids) > 0)
assert allowed[ids].all(), 'Unsupported depth must never enter the browser cloud'
loaded = trimesh.load(scene_path.parent / scene['pointCloudUrl'], force='scene', process=False)
assert len(loaded.geometry) == 1
cloud = next(iter(loaded.geometry.values()))
assert len(cloud.vertices) == scene['pointCloudCount'] == len(ids)
assert np.allclose(cloud.vertices, np.array(scene['points'])[:, 1:], atol=1e-6)
assert np.array_equal(cloud.colors[:, :3], colors[ids]), 'Cloud colors must match original keyframe RGB'
assert scene['complete_room_accepted'] is False
print(json.dumps({'supported_display_points': len(ids), 'exact_source_colors': True,
                  'unsupported_display_points': 0, 'complete_room_accepted': False}))
