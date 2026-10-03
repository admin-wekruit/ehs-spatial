"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_fence_bottom.py."""
import copy
import json
from unittest.mock import patch

import cv2
import numpy as np

from workcell_fence_bottom import fence_bottom_candidates, _join_horizontal, _picket_support


def check():
    # Real LSD detection, paired borders, repeated bar terminations, and a
    # nontrivial raw/canonical pixel-centre affine. No ground height enters.
    rgb = np.full((400, 400, 3), 35, np.uint8)
    for x in (90, 150, 210, 270):
        cv2.rectangle(rgb, (x, 50), (x + 5, 300), (215, 215, 215), -1)
    cv2.rectangle(rgb, (50, 185), (340, 193), (215, 215, 215), -1)
    cv2.rectangle(rgb, (50, 300), (340, 312), (215, 215, 215), -1)
    # A lower floor track with only two posts must not become the chosen rail.
    for x in (65, 325):
        cv2.rectangle(rgb, (x, 313), (x + 6, 350), (215, 215, 215), -1)
    cv2.rectangle(rgb, (50, 350), (340, 360), (215, 215, 215), -1)
    A = np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1.]])
    frame = {'rgb': rgb, 'A': A, 'K': A @ np.array([[400., 0, 200], [0, 400, 200], [0, 0, 1.]]),
             'pose': np.eye(4), 'shape': (200, 200),
             'semanticMasks': {'safety fence': cv2.warpPerspective((rgb[:, :, 0] > 100).astype(np.uint8), A, (200, 200), flags=cv2.INTER_NEAREST)}}
    item = {'id': 'fence-1', 'observations': [{'photo': 2, 'source': 'SAM safety fence intersected with fitted plane 1',
            'polygons': [[[20, 20], [175, 20], [175, 185], [20, 185]]]}]}
    geometry = {'fence': {'planes': [{'normal': [1., 0, 0], 'offset': -8.},
                                      {'normal': [0., 0, 1], 'offset': -4.}]}}
    rows, diagnostics = fence_bottom_candidates(item, geometry, {2: frame}, np.array([0., -1., 0.]))
    assert rows, diagnostics
    strong = [row for row in rows if row['identityEvidence']['independentlySupported']]
    assert strong and all(abs(np.mean(np.asarray(row['rawEnds'])[:, 1]) - 312) < 2 for row in strong), rows
    assert all(row['planeIndex'] == 1 and len(row['identityEvidence']['terminatingPickets']) >= 6 for row in strong)
    assert len(rows) == len({tuple(np.asarray(row['rawSegments']).ravel()) for row in rows})
    assert all(np.allclose(np.asarray(row['uv']), (np.asarray(row['rawEnds']) * .5 - .25)) for row in rows)
    json.dumps({'rows': rows, 'diagnostics': diagnostics}, allow_nan=False)
    # A source-mask rejection cannot be rescued by topology. It must happen
    # before the expensive upright intersections, including with many lines.
    filled = {**frame, 'semanticMasks': {'safety fence': np.ones(frame['shape'], np.uint8)}}
    with patch('workcell_fence_bottom._picket_support', side_effect=AssertionError('Topology ran for an already rejected mask')):
        rejected, skipped = fence_bottom_candidates(item, geometry, {2: filled}, [0, -1, 0])
    assert not rejected and skipped[0]['pairs']
    assert all(pair.get('topologySkipped') and not pair['accepted'] for pair in skipped[0]['pairs'])
    # Depth-to-plane filtering can erase the middle of the rail from catalog
    # polygons. The original SAM mask still contains its face: polygons locate
    # the instance, but must not veto those valid RGB terminal edges.
    sparse = copy.deepcopy(item)
    sparse['observations'][0]['polygons'] = [[[20, 20], [55, 20], [55, 185], [20, 185]],
                                          [[140, 20], [175, 20], [175, 185], [140, 185]]]
    recovered, _ = fence_bottom_candidates(sparse, geometry, {2: frame}, [0, -1, 0])
    recovered_strong = [row for row in recovered if row['identityEvidence']['independentlySupported']]
    assert recovered_strong and all(abs(np.mean(np.asarray(row['rawEnds'])[:, 1]) - 312) < 2 for row in recovered_strong)
    # An occluder separating collinear observed spans cannot delete the spans
    # or make its hidden interval part of the evidence.
    fragments = [{'index': 1, 'raw': np.array([[50., 312], [65., 312]])},
                 {'index': 2, 'raw': np.array([[325., 312], [340., 312]])}]
    joined = _join_horizontal(fragments, frame, np.array([0., 0., 1.]), -4.)
    assert len(joined) == 1 and len(joined[0]['rawSegments']) == 2
    assert abs(joined[0]['lengthRaw'] - 30) < 1e-6
    # Source topology uses the local upper-edge incidence, not its mean world
    # height or a long picket's midpoint. Extruded top-face contact can project
    # above the rail front border; no target ground clearance enters this test.
    lower = np.array([[0., 0., 4.], [4., .4, 4.]])
    upper = lower + [0., .1, 0.]
    vertical = [{'index': i, 'lengthRaw': 500.,
                 'world': np.array([[x, .1 * x + .17, 4.], [x, .1 * x + 2.17, 4.]])}
                for i, x in enumerate(np.linspace(.5, 3.5, 6))]
    supported, evidence = _picket_support(vertical, np.array([1., 0, 0]), np.array([0., 1., 0]),
                                          0., 4., lower, upper, .1)
    assert supported and len(evidence['terminatingPickets']) == 6
    # A middle rail with pickets on both sides is not a lower edge, even if
    # the crop hides all lower rails. Same mask alone does not establish identity.
    altered = rgb.copy(); altered[290:] = 35
    rows, _ = fence_bottom_candidates(item, geometry, {2: {**frame, 'rgb': altered}}, [0, -1, 0])
    assert not any(row['identityEvidence']['independentlySupported'] for row in rows), rows
    partial = rgb.copy()
    for x in (210, 270):
        partial[50:185, x:x + 6] = 35
        partial[194:300, x:x + 6] = 35
    partial_frame = {**frame, 'rgb': partial,
                     'semanticMasks': {'safety fence': cv2.warpPerspective((partial[:, :, 0] > 100).astype(np.uint8), A, (200, 200), flags=cv2.INTER_NEAREST)}}
    partial_rows, _ = fence_bottom_candidates(item, geometry, {2: partial_frame}, [0, -1, 0])
    partial_bottoms = [row for row in partial_rows if abs(np.mean(np.asarray(row['rawEnds'])[:, 1]) - 312) < 2]
    assert partial_bottoms and all(not row['identityEvidence']['independentlySupported'] for row in partial_bottoms)
    reordered = copy.deepcopy(item)
    reordered['geometryPlaneIndex'] = 1
    reordered['observations'][0]['source'] = 'SAM safety fence intersected with fitted plane 0'
    mapped, _ = fence_bottom_candidates(reordered, geometry, {2: frame}, [0, -1, 0])
    assert mapped and all(row['planeIndex'] == 1 for row in mapped)
    invalid = copy.deepcopy(item); invalid['geometryPlaneIndex'] = -1
    try:
        fence_bottom_candidates(invalid, geometry, {2: frame}, [0, -1, 0])
    except ValueError as error:
        assert 'plane index' in str(error)
    else:
        raise AssertionError('Invalid current plane identity must not default to the right-hand plane')
    print('fence lower-rail identity, middle-rail/floor-track rejection, plane identity and pixel affine: OK')


if __name__ == '__main__':
    check()
