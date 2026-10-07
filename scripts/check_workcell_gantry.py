"""Deterministic regression check for workcell_gantry.

* synthetic portal (two posts, a level beam, one brace) seen by two cameras, noisy depth and a 3% depth
  bias in one photo: members, positions, widths, two-photo support and reprojection IoU are recovered;
* render() places a box's front face at its true depth;
* refusals: no segmentation, a single photo, posts without a beam;
* the 2026-10-03 consolidated run with the probe masks, when present: both cells model an entrance portal.

python scripts/check_workcell_gantry.py
"""
import base64
import gzip
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_apps.sam3_app import _encode_coco_rle
from scripts import workcell_gantry as G

H, W = 240, 320
K = np.array([[260., 0, 160], [0, 260., 120], [0, 0, 1]])
GEOMETRY = {'floor': {'normal': [0., -1., 0.], 'offset': 2., 'residualP95Native': .01}}  # camera y down; floor at y = 2
RUN = Path('/Users/adam/Desktop/panoptes-public/research-notes/workcell-oneshot-2026-10-03-a-consolidated')
PROBE = Path('/Users/adam/Desktop/panoptes-public/research-notes/workcell-gantry-2026-10-04/gantry-sam3.json')


def _box(a, b, width, side):
    """(centre, axes, half extents) of a member box between axis points a and b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    side = np.asarray(side, float) - d * (np.asarray(side, float) @ d)
    side /= np.linalg.norm(side)
    return (a + b) / 2, np.column_stack((d, side, np.cross(d, side))), np.array([np.linalg.norm(b - a), width, width]) / 2


TOP = -1.55  # post tops (y); height 3.55 above the floor
TRUTH = {'post-left': _box([-1.4, 2, 6.], [-1.4, TOP, 6.], .12, [1, 0, 0]),
         'post-right': _box([1.4, 2, 6.3], [1.4, TOP, 6.3], .12, [1, 0, 0]),
         'beam': _box([-1.46, TOP - .05, 6.], [1.46, TOP - .05, 6.3], .1, [0, 1, 0]),
         'brace': _box([-1.4, -.6, 6.], [-.6, TOP, 6.09], .1, [0, 0, 1])}
WORDS = {'post-left': 'white steel beam', 'post-right': 'white steel beam', 'brace': 'white steel beam', 'beam': 'blue pipe'}


def _pose(position, yaw_deg):
    t = np.radians(yaw_deg)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]
    pose[:3, 3] = position
    return pose


def _spec(a):
    a = np.ascontiguousarray(a)
    return {'data': base64.b64encode(a.tobytes()).decode(), 'dtype': str(a.dtype), 'shape': list(a.shape)}


def _scene(root, depth_bias=(1., 1.03), seed=0):
    """Two synthetic frames of the portal over a floor and a back wall; per-member SAM masks."""
    rng = np.random.default_rng(seed)
    names = list(TRUTH)
    masks = {}
    for photo, pose, bias in ((1, _pose([0, 0, 0], 0), depth_bias[0]), (2, _pose([1.8, .1, 1.], -19), depth_bias[1])):
        frame = {'valid': np.ones((H, W), bool), 'K': K, 'pose': pose}
        owner, depth = G.render([TRUTH[n] for n in names], frame)
        yy, xx = np.mgrid[0:H, 0:W]
        rays = np.stack([xx, yy, np.ones_like(xx)], -1).astype(float) @ np.linalg.inv(K).T @ pose[:3, :3].T
        background = np.minimum(np.where(rays[..., 1] > 1e-6, (2 - pose[1, 3]) / np.maximum(rays[..., 1], 1e-6), np.inf),
                                (12 - pose[2, 3]) / rays[..., 2])  # floor y = 2, wall z = 12 (camera z component 1)
        depth = np.where(owner >= 0, depth * bias * (1 + rng.normal(0, .005, depth.shape)), background)
        points = pose[:3, 3] + rays * depth[..., None]
        raw = {'image': _spec(np.full((H, W, 3), 128, np.uint8)), 'pts3d': _spec(points.astype(np.float32)),
               'non_ambiguous_mask': _spec(np.ones((H, W), bool)), 'camera_poses': _spec(pose.astype(np.float32)),
               'intrinsics': _spec(K.astype(np.float32))}
        with gzip.open(root / f'frame_{photo:04d}.json.gz', 'wt') as stream:
            json.dump(raw, stream)
        masks[photo] = {}
        for n, name in enumerate(names):
            if (owner == n).any():
                masks[photo].setdefault(WORDS[name], []).append(_encode_coco_rle(owner == n))
    return masks


def check_render():
    box = (np.array([0., 0, 5]), np.eye(3), np.array([.5, .5, .5]))
    owner, depth = G.render([box], {'valid': np.ones((H, W), bool), 'K': K, 'pose': np.eye(4)})
    assert owner[120, 160] == 0 and abs(depth[120, 160] - 4.5) < 1e-9, (owner[120, 160], depth[120, 160])
    assert owner[0, 0] == -1 and not np.isfinite(depth[0, 0])
    expected = 2 * int(np.floor(.5 / 4.5 * 260)) + 1  # front-face silhouette width in pixels through the centre row
    assert abs(int((owner[120] == 0).sum()) - expected) <= 1, ((owner[120] == 0).sum(), expected)


def check_synthetic():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        masks = _scene(root)
        scene, record = G.build_gantry(root, [1, 2], masks, GEOMETRY)
        assert scene is not None and record['status'] == 'modelled', record.get('reason')
        types = sorted(m['type'] for m in record['members'])
        assert types == ['beam', 'brace', 'post', 'post'], types
        up = np.array([0., -1, 0])
        for name in ('post-left', 'post-right'):
            centre, _, half = TRUTH[name]
            post = min((m for m in record['members'] if m['type'] == 'post'),
                       key=lambda m: np.linalg.norm(np.asarray(m['endsNative'][0])[[0, 2]] - centre[[0, 2]]))
            foot, head = np.asarray(post['endsNative'])
            assert np.linalg.norm(foot[[0, 2]] - centre[[0, 2]]) < .05, (name, foot)
            assert abs(foot @ up + 2) < 1e-6, 'posts start at the fitted floor'
            assert abs(head @ up + 2 - (2 - TOP + .05)) < .08, ('post rises to the top of its beam', head @ up + 2)
            assert abs(post['widthNative'] - 2 * half[1]) < .3 * 2 * half[1], (name, post['widthNative'])
            assert post['depthFrom'].startswith('triangulated'), 'two photos fix the axis despite the 3% depth bias'
        beam = next(m for m in record['members'] if m['type'] == 'beam')
        assert abs(beam['heightsNative'][0] - (2 - TOP + .05)) < .05, beam['heightsNative']
        assert abs(beam['lengthNative'] - 2 * TRUTH['beam'][2][0]) < .25, beam['lengthNative']
        brace = next(m for m in record['members'] if m['type'] == 'brace')
        assert 30 < brace['angleFromVerticalDeg'] < 50, brace['angleFromVerticalDeg']
        assert all(m['sourcePhotos'] == [1, 2] and not m['singleView'] for m in record['members']), \
            [(m['id'], m['supportByPhoto']) for m in record['members']]
        assert {j['kind'] for j in record['joints']} == {'carries', 'brace end'}
        for p, v in record['validation'].items():
            assert v['iou'] > .6 and v['depth']['medianRelative'] < .05, (p, v['iou'], v['depth'])  # photo 2 depth is 3% biased
        sources = {(o['photo'], o['source']) for o in record['observations']}
        assert (1, 'SAM: white steel beam; instance 0') in sources and (2, 'SAM: blue pipe; instance 0') in sources, sources
        assert all(o['polygons'] for o in record['observations'])
        canonical = [(o['photo'], o['source'].rsplit('; instance ', 1)[0], int(o['source'].rsplit('; instance ', 1)[1])) for o in record['observations']]
        assert canonical == sorted(canonical), canonical  # order never follows float-sensitive member support
        json.dumps(record, allow_nan=False)
        assert len(scene.geometry) == 4
        again = G.build_gantry(root, [1, 2], masks, GEOMETRY)[1]
        assert json.dumps(again, sort_keys=True) == json.dumps(record, sort_keys=True), 'deterministic'
        print('synthetic', {m['id']: (m['lengthNative'], m['widthNative']) for m in record['members']},
              {p: v['iou'] for p, v in record['validation'].items()})
        # refusals: missing evidence never yields a model
        for photos, given, reason in (([1, 2], {}, 'No gantry segmentation'), ([1], masks, 'at least two photos'),
                                      ([1, 2], {p: {'white steel beam': m['white steel beam'][:2]} for p, m in masks.items()},
                                       'No beam carried by a post')):
            scene, refusal = G.build_gantry(root, photos, given, GEOMETRY)
            assert scene is None and refusal['status'] == 'refused' and reason in refusal['reason'], refusal


def check_run():
    if not (RUN / 'frame_0004.json.gz').is_file() or not PROBE.is_file():
        print('real run absent; skipped')
        return
    geometry = json.loads((RUN / 'geometry.json').read_text())
    segmentation = json.loads(PROBE.read_text())
    for photos in ([3, 4], [1, 2]):
        scene, record = G.build_gantry(RUN, photos, G.masks_from_segmentation(segmentation, photos), geometry)
        assert scene is not None, record
        entrance = [f for f in record['frames'] if any(o['source'].startswith('SAM: blue pipe') for m in record['members']
                                                       if m['id'] in f for o in m['observations'])]
        assert entrance and sum(m['type'] == 'post' for m in record['members'] if m['id'] in entrance[0]) == 2, record['frames']
        anchoring = {r['objectId']: r for r in G.proxy_anchoring(RUN, record, G._frames(RUN, photos), geometry)}
        lamp = 'signal-light-2' if photos == [3, 4] else 'signal-light-4'
        assert anchoring[lamp]['proxy']['gapNative'] <= .15, anchoring[lamp]
        print('run photos', photos, [(m['id'], m['sourcePhotos']) for m in record['members']],
              {p: v['iou'] for p, v in record['validation'].items()})


if __name__ == '__main__':
    check_render()
    check_synthetic()
    check_run()
    print('check_workcell_gantry passed')
