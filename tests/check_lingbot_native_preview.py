"""Read exported diagnostics independently; failed checks must remain failed in the viewer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reconstruct_room_rgb import digest

p = argparse.ArgumentParser(description=__doc__)
for name in ['scene', 'run', 'review']:
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
s = json.loads(a.scene.read_text()); execution = json.loads((a.run / 'run.json').read_text())
plan = json.loads((a.run / 'plan.json').read_text()); review = json.loads(a.review.read_text())
assert s['complete_room_accepted'] is False and s['units'] == 'uncalibrated_monocular'
assert s['provenance']['temporal_geometry_check'] == review
assert s['provenance']['execution_sha256'] == digest(a.run / 'run.json') == review['execution_sha256']
assert s['source_video_sha256'] == plan['source_video_sha256']
assert not s.get('meshUrl') and not s.get('bodyKeyframes')
cloud = a.scene.parent / s['pointCloudUrl']
assert digest(cloud) == s['provenance']['point_cloud_sha256']
loaded = trimesh.load(cloud, force='scene', process=False)
assert sum(len(g.vertices) for g in loaded.geometry.values()) == s['pointCloudCount'] <= 300000
for f, native, source, contract in zip(s['frames'], execution['frames'], plan['frames'], s['provenance']['geometry_contract_per_frame'], strict=True):
    assert f['sourceFrame'] == native['sourceFrame'] == source['sourceFrame'] == contract['sourceFrame']
    assert f['timeSec'] == source['timeSec'] and not f['objects']
    with np.load(a.run / native['file'], allow_pickle=False) as data:
        inverse = np.eye(4); inverse[:3] = data['w2c']
        assert np.allclose(np.asarray(f['c2w']) @ inverse, np.eye(4), atol=1e-7)
    assert contract['pointmap_xyz_reprojection_error_p95_px'] < .001
assert all(x['endTimeSec'] == y['timeSec'] for x, y in zip(s['frames'], s['frames'][1:]))
print(json.dumps({'native_frames_checked': len(s['frames']), 'display_points': s['pointCloudCount'],
                  'original_review_preserved': True, 'complete_room_accepted': False, 'scene_sha256': digest(a.scene)}))
