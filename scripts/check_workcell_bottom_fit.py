"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_bottom_fit.py."""
import copy
import json

import numpy as np

from workcell_bottom_fit import fit_bottoms, leave_one_photo_out
from workcell_photo_metrology import _pixels
from workcell_photo_objects import _project


def check():
    normal = np.array([.15, -.1, 1.]); normal /= np.linalg.norm(normal)
    u = np.cross(normal, [1., 0., 0.]); u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(normal, u)))
    origin = np.array([.3, -.2, .15])
    ground = {'normal': (normal * 3).tolist(), 'offset': float(-3 * normal @ origin)}
    frames = {}
    for photo, location in enumerate(([-3, -4, 2.8], [4, -3, 3], [-4, 3, 2.6], [3, 4, 3.4]), 1):
        eye = origin + basis @ location[:2] + normal * location[2]
        forward = origin + normal * .4 - eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward, normal); right /= np.linalg.norm(right)
        pose = np.eye(4); pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward)); pose[:3, 3] = eye
        frames[photo] = {'pose': pose, 'K': np.array([[650., 0, 400], [0, 610, 300], [0, 0, 1]]),
                         'A': np.array([[.6, 0, -31.], [0, .55, -24.], [0, 0, 1]])}

    def inputs(heights):
        items, truths = [], []
        angle = .33
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for index, height in enumerate(heights):
            center = origin + basis @ [-.65 + index * 1.3, index * .2]
            local = np.array([[-.13, -.09], [.13, -.09], [.13, .09], [-.13, .09]]) @ rotation.T
            truth = center + local @ basis.T + height * normal
            original = truth + basis @ [.035, -.027] + normal * (.13 - index * .04)
            observations = []
            for photo, frame in frames.items():
                projected, depth = _project(truth, frame)
                assert (depth > 0).all()
                raw = _pixels(projected, np.linalg.inv(frame['A']))
                edge = (photo + index) % 4
                start, stop = raw[edge], raw[(edge + 1) % 4]
                segments = np.array([[start + (stop - start) * .12, start + (stop - start) * .4],
                                     [start + (stop - start) * .61, start + (stop - start) * .86]])
                wrong = segments + np.array([27 * (-1) ** photo, 34 + 11 * photo])
                observations.extend([
                    {'photo': photo, 'id': f'wrong-{photo}', 'rawSegments': wrong.tolist(), 'score': .99},
                    {'photo': photo, 'id': f'correct-{photo}', 'rawSegments': segments.tolist(),
                     # Deliberately wrong envelope: observed rawSegments must win.
                     'rawEnds': (np.array([start, stop]) + 400).tolist(), 'score': .1}])
            # Non-cyclic input order must round-trip for caller mesh vertex indices.
            order = [2, 0, 3, 1]
            items.append({'id': f'post-{index}', 'bottomVerticesNative': original[order].tolist(), 'observations': observations})
            truths.append(truth[order])
        return items, truths

    for shared, heights in ((True, [.43, .43]), (False, [.32, .59])):
        items, truths = inputs(heights)
        original_inputs = copy.deepcopy(items)
        result = fit_bottoms(items, frames, ground, shared, max_starts=8)
        assert result['status'] == 'conditional_fit', result['optimizer']
        assert result['mPerNative'] is None and result['physicalValidation'] == 'none'
        assert result['optimizer']['evaluatedStarts'] > 1
        assert result['optimizer']['rmsRawPx'] < 1e-5, result['optimizer']
        assert items == original_inputs, 'Fitting must not mutate source inputs'
        for index, row in enumerate(result['items']):
            assert np.allclose(row['bottomVerticesNative'], truths[index], atol=1e-5), row
            assert abs(row['heightNative'] - heights[index]) < 1e-5
            assert all(view['selectedCandidateId'].startswith('correct-') for view in row['views'])
            assert all(view['maxRawPx'] < 1e-5 for view in row['views'])
            assert max(view['rmsRawPx'] for view in row['beforeViews']) > 1
            assert np.allclose(np.array(row['originalBottomVerticesNative']) + row['translationNative'], row['bottomVerticesNative'])
        if shared:
            assert result['items'][0]['heightNative'] == result['items'][1]['heightNative'] == result['sharedHeightNative']
        else:
            assert result['items'][1]['heightNative'] - result['items'][0]['heightNative'] > .2
        json.dumps(result, allow_nan=False)

    items, _ = inputs([.43, .43])
    holdout = leave_one_photo_out(items, frames, ground, True, max_starts=4)
    assert len(holdout['folds']) == 4
    for fold in holdout['folds']:
        assert fold['status'] == 'conditional_fit', fold
        for row in fold['heldOut']:
            assert row['view']['rmsRawPx'] < 1e-4, row
        assert all(view['photo'] != fold['photo'] for item in fold['fit']['items'] for view in item['views'])
    json.dumps(holdout, allow_nan=False)

    # Counterexample: holding out the only identity witness must not turn a
    # precise weak+weak projection fit into an identity-supported result.
    anchored, truths = inputs([.43])
    anchored[0].update(requireIdentityAnchor=True, bottomVerticesNative=truths[0].tolist())
    for row in anchored[0]['observations']:
        row['identityEvidence'] = {'independentlySupported': False}
        row['partIdentity'] = {'frontOrSide': 'unresolved', 'commonBottomPlaneSupported': False}
    for function in (fit_bottoms, leave_one_photo_out):
        try:
            function(anchored, frames, ground, max_starts=1)
        except ValueError as error:
            assert 'identity anchor' in str(error), str(error)
        else:
            raise AssertionError('A required identity anchor was absent from all source views')
    witness = next(row for row in anchored[0]['observations'] if row['id'] == 'correct-1')
    witness['identityEvidence']['independentlySupported'] = True
    result = fit_bottoms(anchored, frames, ground, max_starts=1)
    assert result['status'] == 'conditional_fit'
    assert result['items'][0]['identityAnchorPhotos'] == [1]
    selected = result['items'][0]['views'][0]
    assert selected['identityStatus'] == 'source_anchor'
    assert selected['identityEvidence'] == witness['identityEvidence']
    assert selected['partIdentity'] == witness['partIdentity']
    assert result['physicalValidation'] == 'none'
    folds = leave_one_photo_out(anchored, frames, ground, max_starts=1)
    lost = next(row for row in folds['folds'] if row['photo'] == 1)
    assert lost['status'] == 'unsupported' and 'identity anchor' in lost['reason']
    assert 'fit' not in lost
    assert all(row['status'] == 'conditional_fit' for row in folds['folds'] if row['photo'] != 1)
    # A strong candidate merely being present is insufficient if optimization
    # selects a different weak candidate in that photo.
    witness['identityEvidence']['independentlySupported'] = False
    next(row for row in anchored[0]['observations'] if row['id'] == 'wrong-1')['identityEvidence']['independentlySupported'] = True
    rejected = fit_bottoms(anchored, frames, ground, max_starts=1)
    assert rejected['optimizer']['rmsRawPx'] < 1e-5
    assert rejected['status'] == 'unsupported'
    assert not rejected['items'][0]['identityAnchorRequirementSatisfied']
    assert rejected['items'][0]['identityAnchorPhotos'] == []
    json.dumps([result, folds, rejected], allow_nan=False)

    # Four different physical rectangle sides can each fit perfectly when a
    # view chooses its own model edge. An explicit same-edge hypothesis cannot.
    switching, truths = inputs([.43])
    switching[0]['bottomVerticesNative'] = truths[0].tolist()
    free = fit_bottoms(switching, frames, ground, max_starts=1)
    assert free['optimizer']['rmsRawPx'] < 1e-5
    assert len({tuple(v['modelEdgeVertexIndices']) for v in free['items'][0]['views']}) > 1
    switching[0]['samePhysicalEdge'] = True
    constrained = fit_bottoms(switching, frames, ground, max_starts=4)
    row = constrained['items'][0]
    assert len({tuple(v['modelEdgeVertexIndices']) for v in row['views']}) == 1, row['views']
    assert constrained['optimizer']['rmsRawPx'] > .1
    assert len(row['edgeHypotheses']) == 4
    assert all(len(v['candidates']) == 2 for h in row['edgeHypotheses'] for v in h['views'])
    held = leave_one_photo_out(switching, frames, ground, max_starts=1)
    for fold in held['folds']:
        expected = fold['fit']['items'][0]['selectedModelEdgeVertexIndices']
        assert fold['heldOut'][0]['view']['modelEdgeVertexIndices'] == expected
    # A consistent edge retains its exact fit under tilted ground and affine
    # camera pixels; the constraint is a hypothesis, not physical validation.
    consistent = copy.deepcopy(switching)
    for observation in consistent[0]['observations']:
        uv, _ = _project(truths[0][[0, 2]], frames[observation['photo']])
        raw = _pixels(uv, np.linalg.inv(frames[observation['photo']]['A']))
        observation['rawSegments'] = [raw.tolist()]
    exact = fit_bottoms(consistent, frames, ground, max_starts=1)
    assert exact['optimizer']['rmsRawPx'] < 1e-5
    assert exact['physicalValidation'] == 'none'
    assert len({tuple(v['modelEdgeVertexIndices']) for v in exact['items'][0]['views']}) == 1
    json.dumps([constrained, held, exact], allow_nan=False)

    single = [{**items[0], 'observations': [row for row in items[0]['observations'] if row['photo'] == 1]}]
    try:
        fit_bottoms(single, frames, ground)
    except ValueError as error:
        assert 'two source photos' in str(error)
    else:
        raise AssertionError('Accepted a single source view')
    # Repeated camera plus identical line cannot determine all three translations.
    degenerate = copy.deepcopy(single[0])
    degenerate['observations'] += [{**row, 'photo': 2} for row in degenerate['observations']]
    result = fit_bottoms([degenerate], {1: frames[1], 2: frames[1]}, ground, max_starts=2)
    assert result['status'] == 'unsupported' and result['optimizer']['jacobianRank'] < 3
    print('PASS: tilted ground and perspective/affine cameras; shared and independent native heights; wrong higher-score candidates rejected; visible fragments only; input vertex order; finite JSON; leave-one-photo-out; required/selected identity anchors and lost-witness rejection; joint same-edge selection and held-out edge preservation; single-view and rank failure.')


if __name__ == '__main__':
    check()
